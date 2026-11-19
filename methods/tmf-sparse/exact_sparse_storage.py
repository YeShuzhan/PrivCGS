"""Sparse execution of the unchanged TmF uint8 matrix operations.

No sampler is replaced. Unknown matrix operations fail instead of densifying.
"""
import ast, hashlib, inspect, json, pickle
from pathlib import Path
import numpy as np
from scipy import sparse
import networkx as nx


class Mask:
    def __init__(self, matrix): self.matrix=matrix


class Matrix:
    __array_priority__=10000
    ndim=2
    dtype=np.dtype(np.uint8)

    def __init__(self,n,csr=None):
        self.n=int(n);self.shape=(self.n,self.n)
        self.entries={} if csr is None else None
        self.csr=None if csr is None else csr.astype(np.uint8).tocsr()
        if self.csr is not None:
            self.csr.sum_duplicates();self.csr.eliminate_zeros();self.csr.sort_indices()

    def __len__(self): return self.n
    def __array__(self,*a,**kw): raise TypeError('Dense conversion is forbidden')

    def as_csr(self):
        if self.csr is None:
            size=len(self.entries)
            keys=np.fromiter(self.entries.keys(),dtype=np.int64,count=size)
            values=np.fromiter(self.entries.values(),dtype=np.uint8,count=size)
            self.csr=sparse.csr_matrix((values,(keys//self.n,keys%self.n)),shape=self.shape,dtype=np.uint8)
            self.csr.sort_indices();self.entries=None
        return self.csr

    def __getitem__(self,key):
        assert isinstance(key,tuple) and len(key)==2
        u,v=map(int,key);assert 0<=u<self.n and 0<=v<self.n
        return np.uint8(self.entries.get(u*self.n+v,0)) if self.entries is not None else self.csr[u,v]

    def __setitem__(self,key,value):
        if isinstance(key,Mask):
            assert key.matrix is self and int(value)==1
            if self.entries is not None:
                for k in self.entries:self.entries[k]=1
            else:self.csr.data[:]=1
            return
        assert isinstance(key,tuple) and len(key)==2
        u,v=map(int,key);assert 0<=u<self.n and 0<=v<self.n
        value=int(value);assert 0<=value<=255
        if self.entries is not None:
            k=u*self.n+v
            if value:self.entries[k]=value
            else:self.entries.pop(k,None)
        else:
            self.csr[u,v]=np.uint8(value)

    def __add__(self,other):
        assert isinstance(other,Matrix) and other.shape==self.shape
        return Matrix(self.n,self.as_csr()+other.as_csr())

    def __gt__(self,value):assert value==0;return Mask(self)
    def __ne__(self,value):assert value==0;return Mask(self)

    def nonzero(self):
        a,b=self.as_csr().nonzero()
        return np.asarray(a,dtype=np.intp),np.asarray(b,dtype=np.intp)


def digest(v):return hashlib.sha256(pickle.dumps(v,protocol=4)).hexdigest()


def install(legacy_root,mode,trace):
    root=Path(legacy_root).resolve();main=root/'code/author_zip_zcdp/main_TmF_zcdp.py';utils=root/'code/author_zip_zcdp/utils_zcdp.py'
    assert mode in ('reference','sparse')
    tree=ast.parse(main.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main_func_TmF')
    target=next(x for x in ast.walk(fn) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mat1' for t in x.targets))
    matrix_runtime_line=target.lineno-fn.lineno+1
    old_zero=np.zeros;old_transpose=np.transpose;old_triu=np.triu;old_nonzero=np.nonzero;old_where=np.where;old_from=nx.from_numpy_array

    def zeros(shape,*args,**kwargs):
        caller=inspect.currentframe().f_back
        path=Path(caller.f_code.co_filename).resolve()
        is_loader=path==utils and caller.f_code.co_name=='get_mat' and caller.f_lineno==45
        is_output=path==main and caller.f_code.co_name=='main_func_TmF' and caller.f_code.co_firstlineno==1 and caller.f_lineno==matrix_runtime_line
        square=isinstance(shape,(tuple,list)) and len(shape)==2 and shape[0]==shape[1]
        if is_loader or is_output:
            assert square and not args and kwargs=={'dtype':np.uint8}
            n=int(shape[0]);trace['matrix_allocations'].append(dict(site='input' if is_loader else 'output',shape=[n,n],dtype='uint8',dense_bytes=n*n,sparse=mode=='sparse'))
            if mode=='sparse':return Matrix(n)
            assert n<=256,'Reference dense execution is limited to authorized Polbooks tests'
        elif square and int(shape[0])>256:
            raise AssertionError('Unrecognized dense square allocation')
        return old_zero(shape,*args,**kwargs)

    def transpose(a,axes=None):
        if isinstance(a,Matrix):
            assert axes is None;return Matrix(a.n,a.as_csr().transpose().tocsr())
        return old_transpose(a,axes)

    def upper(a,k=0):
        if isinstance(a,Matrix):
            assert k==1;return Matrix(a.n,sparse.triu(a.as_csr(),k=1,format='csr'))
        return old_triu(a,k)

    def nonzero(a):
        if isinstance(a,Matrix):return a.nonzero()
        return old_nonzero(a)

    def where(condition,*args):
        if isinstance(condition,Mask):assert not args;return condition.matrix.nonzero()
        return old_where(condition,*args)

    def graph(a,parallel_edges=False,create_using=None,edge_attr='weight'):
        if isinstance(a,Matrix):
            assert not parallel_edges and create_using in (None,nx.Graph) and edge_attr=='weight'
            g=nx.empty_graph(0,create_using);g.add_nodes_from(range(a.n))
            csr=a.as_csr()
            # Same row-major nonzero order and Python integer weights as from_numpy_array(uint8).
            for u in range(a.n):
                for k in range(csr.indptr[u],csr.indptr[u+1]):g.add_edge(u,int(csr.indices[k]),weight=int(csr.data[k]))
        else:g=old_from(a,parallel_edges=parallel_edges,create_using=create_using,edge_attr=edge_attr)
        trace['original_graph_order_hash']=digest((list(g.nodes()),[(u,list(g[u].items())) for u in g]))
        trace['original_graph_nodes']=g.number_of_nodes();trace['original_graph_edges']=g.number_of_edges()
        return g

    np.zeros=zeros;np.transpose=transpose;np.triu=upper;np.nonzero=nonzero;np.where=where;nx.from_numpy_array=graph
    return dict(matrix_runtime_line=matrix_runtime_line,main_source_sha256=hashlib.sha256(main.read_bytes()).hexdigest(),utils_source_sha256=hashlib.sha256(utils.read_bytes()).hexdigest())
