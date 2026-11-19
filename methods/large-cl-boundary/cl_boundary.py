"""Frozen NetworkX body; repair only the undefined rounded-to-one log base."""
import ast,hashlib,math
from fractions import Fraction
from operator import itemgetter
import networkx as nx
EXPECTED_NETWORKX_SHA='d19a6fe2ae6b620ad47a9a6b386690153c373d583a1045a86e8f62dc8adaf592'

def geometric_skip(r,p,remaining):
    base=1-p
    if base!=1.0:
        return math.floor(math.log(r,base))
    assert p>0 and remaining>0
    if r<=0:return math.floor(math.log(r,base))
    assert r<=1
    pf=Fraction.from_float(float(p));rf=Fraction.from_float(float(r))
    # Bernoulli inequality proves the skip leaves the remaining candidate range.
    if 1-rf>=remaining*pf:return remaining
    exact_base=1-pf
    if rf<=exact_base**remaining:return remaining
    lo,hi=0,remaining
    while lo+1<hi:
        middle=(lo+hi)//2
        if rf<=exact_base**middle:lo=middle
        else:hi=middle
    return lo

def build(path):
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED_NETWORKX_SHA
    tree=ast.parse(raw);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='expected_degree_graph')
    fn.name='_expected_degree_body';fn.decorator_list=[];changed=[]
    class Boundary(ast.NodeTransformer):
        def visit_AugAssign(self,node):
            if isinstance(node.target,ast.Name) and node.target.id=='v' and ast.unparse(node.value)=='math.floor(math.log(r, 1 - p))':
                node.value=ast.parse('_geometric_skip(r,p,n-v)',mode='eval').body;changed.append(node.lineno)
            return node
    Boundary().visit(fn);assert changed==[432],changed
    scope=dict(nx=nx,math=math,itemgetter=itemgetter,_geometric_skip=geometric_skip)
    module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module)
    exec(compile(module,__file__+':independent_frozen_body','exec'),scope)
    return nx.utils.py_random_state(1)(scope['_expected_degree_body'])
