// Exact legal-ID rank/select. No complement table or proposal retry.
struct LegalIndex {
    struct Node { int key,height; int64_t left,right,size; };
    int n;int64_t used=0;
    std::vector<int64_t> head,next,root;
    std::vector<int> neighbor;
    std::vector<Node> nodes;
    int64_t activations=0,rank_queries=0;
    LegalIndex(int n_,int64_t m):n(n_),head(n,-1),next(2*m),root(n,-1),neighbor(2*m){}
    int h(int64_t x)const{return x<0?0:nodes[x].height;}
    int64_t size(int64_t x)const{return x<0?0:nodes[x].size;}
    void fix(int64_t x){nodes[x].height=1+std::max(h(nodes[x].left),h(nodes[x].right));nodes[x].size=1+size(nodes[x].left)+size(nodes[x].right);}
    int64_t right(int64_t y){auto x=nodes[y].left,b=nodes[x].right;nodes[x].right=y;nodes[y].left=b;fix(y);fix(x);return x;}
    int64_t left(int64_t x){auto y=nodes[x].right,b=nodes[y].left;nodes[y].left=x;nodes[x].right=b;fix(x);fix(y);return y;}
    int64_t insert(int64_t x,int key){
        if(x<0){nodes.push_back(Node{key,1,-1,-1,1});return int64_t(nodes.size()-1);}
        if(key<nodes[x].key){auto z=insert(nodes[x].left,key);nodes[x].left=z;}
        else if(key>nodes[x].key){auto z=insert(nodes[x].right,key);nodes[x].right=z;}else return x;
        fix(x);int b=h(nodes[x].left)-h(nodes[x].right);
        if(b>1){if(key>nodes[nodes[x].left].key)nodes[x].left=left(nodes[x].left);return right(x);}
        if(b< -1){if(key<nodes[nodes[x].right].key)nodes[x].right=right(nodes[x].right);return left(x);}return x;
    }
    void activate(int u){if(root[u]>=0)return;root[u]=insert(-1,u);for(auto i=head[u];i>=0;i=next[i])root[u]=insert(root[u],neighbor[i]);activations++;}
    int64_t rank(int64_t x,int value){int64_t ans=0;while(x>=0){if(nodes[x].key<=value){ans+=1+size(nodes[x].left);x=nodes[x].right;}else x=nodes[x].left;}rank_queries++;return ans;}
    int select(int u,uint64_t k){activate(u);auto legal=int64_t(n)-size(root[u]);if(k>=uint64_t(legal))throw std::runtime_error("legal rank outside domain");int lo=0,hi=n-1;while(lo<hi){int mid=lo+(hi-lo)/2;if(uint64_t(int64_t(mid)+1-rank(root[u],mid))>k)hi=mid;else lo=mid+1;}return lo;}
    void add(int u,int v){
        if(root[u]>=0)root[u]=insert(root[u],v);if(root[v]>=0)root[v]=insert(root[v],u);
        neighbor[used]=v;next[used]=head[u];head[u]=used++;
        neighbor[used]=u;next[used]=head[v];head[v]=used++;
    }
    int64_t bytes()const{return int64_t((head.capacity()+next.capacity()+root.capacity())*sizeof(int64_t)+neighbor.capacity()*sizeof(int)+nodes.capacity()*sizeof(Node));}
};
struct MissingTree {
    int n;uint64_t total=0;std::vector<uint64_t> tree;
    MissingTree(int n_,const std::vector<int>& d):n(n_),tree(n_+1,0){for(int i=0;i<n;i++){auto mass=uint64_t(n-1-d[i]);total+=mass;tree[i+1]+=mass;int j=(i+1)+((i+1)&-(i+1));if(j<=n)tree[j]+=tree[i+1];}}
    void remove_one(int u){--total;for(int i=u+1;i<=n;i+=i&-i)--tree[i];}
    int select(uint64_t k)const{if(k>=total)throw std::runtime_error("missing source rank outside domain");int at=0,step=1;while(step<=n/2)step*=2;for(;step;step/=2){int j=at+step;if(j<=n&&tree[j]<=k){k-=tree[j];at=j;}}return at;}
};
