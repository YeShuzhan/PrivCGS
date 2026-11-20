#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

using I = int32_t;
using J = int64_t;

// Prefix maximum at the reversed a coordinate: query(a) finds max c for
// witnesses with at least a common neighbors.
struct Witness {
    std::vector<I> t;
    const I max_degree;
    explicit Witness(I d) : t(d+2,-1), max_degree(d) {}
    I query(I a) const {
        I result=-1;
        for(I i=max_degree-a+1;i>0;i-=i&-i) result=std::max(result,t[i]);
        return result;
    }
    void insert(I a,I c) {
        for(I i=max_degree-a+1;i<(I)t.size();i+=i&-i) t[i]=std::max(t[i],c);
    }
};

static I intersect(const I* a,const I* ae,const I* b,const I* be) {
    if(ae-a>be-b) { std::swap(a,b); std::swap(ae,be); }
    I result=0;
    if(be-b>8*(ae-a)) {
        for(;a<ae && b<be;++a) {
            b=std::lower_bound(b,be,*a);
            if(b<be && *b==*a) { ++result; ++b; }
        }
    } else {
        while(a<ae && b<be) {
            if(*a<*b) ++a;
            else if(*a>*b) ++b;
            else { ++result; ++a; ++b; }
        }
    }
    return result;
}

// Input canonical undirected pairs u < v; output best common-neighbor
// count at each b, ignoring only globally dominated witnesses.
extern "C" int profile(I n,J m,const I* edges,I* best,J* evidence) {
    if(n<3 || m<0) return 1;
    std::vector<I> degrees(n,0), order(n);
    for(J k=0;k<m;++k) {
        const I u=edges[2*k],v=edges[2*k+1];
        if(u<0 || v>=n || u>=v) return 2;
        ++degrees[u]; ++degrees[v];
    }
    std::vector<J> row(n+1,0);
    for(I u=0;u<n;++u) row[u+1]=row[u]+degrees[u];
    std::vector<I> adj(std::max<J>(1,2*m));
    {
        auto end=row;
        for(J k=0;k<m;++k) {
            const I u=edges[2*k],v=edges[2*k+1];
            adj[end[u]++]=v; adj[end[v]++]=u;
        }
    }
    for(I u=0;u<n;++u) {
        auto a=adj.begin()+row[u],e=adj.begin()+row[u+1];
        std::sort(a,e);
        if(std::adjacent_find(a,e)!=e) return 3;
    }
    std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](I u,I v) {
        return degrees[u]!=degrees[v] ? degrees[u]>degrees[v] : u<v;
    });
    Witness witnesses(degrees[order[0]]);
    J checked=0, accepted=0;
    for(I x=0;x+1<n;++x) {
        I u=order[x], du=degrees[u];
        if(witnesses.query(degrees[order[x+1]])>=du+degrees[order[x+1]]) break;
        auto ua=adj.data()+row[u], ue=adj.data()+row[u+1];
        for(I y=x+1;y<n;++y) {
            I v=order[y], dv=degrees[v];
            const I observed=witnesses.query(dv);
            if(observed>=du+dv) break;
            auto va=adj.data()+row[v], ve=adj.data()+row[v+1];
            const I c=du+dv-2*std::binary_search(va,ve,u);
            if(observed>=c) continue;
            const I a=intersect(ua,ue,va,ve);
            ++checked;
            if(witnesses.query(a)>=c) continue;
            witnesses.insert(a,c);
            const I b=c-2*a;
            if(b<0 || b>n-2) return 4;
            best[b]=std::max(best[b],a);
            ++accepted;
        }
    }
    // Independent exact triangle count: orient toward greater degree (ID
    // tie break as in the ordering above) and count each triangle once.
    std::vector<J> frow(n+1,0);
    std::vector<I> forward; forward.reserve(m);
    for(I u=0;u<n;++u) {
        for(J i=row[u];i<row[u+1];++i) {
            I v=adj[i];
            if(degrees[v]>degrees[u] || (degrees[v]==degrees[u] && v<u))
                forward.push_back(v);
        }
        frow[u+1]=forward.size();
    }
    J tri=0;
    if(!forward.empty()) {
        auto p=forward.data();
        for(I u=0;u<n;++u) for(J i=frow[u];i<frow[u+1];++i) {
            I v=forward[i];
            tri+=intersect(p+frow[u],p+frow[u+1],p+frow[v],p+frow[v+1]);
        }
    }
    evidence[0]=tri; evidence[1]=checked; evidence[2]=accepted;
    return 0;
}
