#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <vector>
using I=std::int64_t;using K=std::uint64_t;
static K key(int u,int v){if(u>v)std::swap(u,v);return (K(std::uint32_t(u))<<32)|std::uint32_t(v);}
static void push(std::vector<std::pair<int,int>>&e,int u,int v){if(u>v)std::swap(u,v);e.emplace_back(u,v);}
static void export_edges(std::vector<std::pair<int,int>>&e,I*out){std::sort(e.begin(),e.end());for(std::size_t i=0;i<e.size();i++){out[2*i]=e[i].first;out[2*i+1]=e[i].second;}}

extern "C" __declspec(dllexport) I partial_section(int n,const I*target,I m,const I*source,I nr,const I*remove,I na,const I*add,I*out,I*residual,I*closed,I*laid_off){
 try{
    std::vector<I>d(target,target+n);I maximum=*std::max_element(d.begin(),d.end());
    std::vector<std::set<int>>buckets(maximum+1);std::set<I>levels;
    for(int u=0;u<n;u++){closed[u]=0;if(d[u]>0){buckets[d[u]].insert(u);levels.insert(d[u]);}}
    std::vector<std::vector<int>>adj(n),extra(n);std::unordered_set<K>removed;removed.reserve(nr*2+1);
    for(I j=0;j<m;j++){int u=source[2*j],v=source[2*j+1];adj[u].push_back(v);adj[v].push_back(u);}
    for(I j=0;j<nr;j++)removed.insert(key(remove[2*j],remove[2*j+1]));
    for(I j=0;j<na;j++){int u=add[2*j],v=add[2*j+1];extra[u].push_back(v);extra[v].push_back(u);}
    for(auto&row:adj)std::sort(row.begin(),row.end());
    for(auto&row:extra)std::sort(row.begin(),row.end());
    std::vector<std::pair<int,int>>generated;I total=0;for(I v:d)total+=v;generated.reserve(total/2);*laid_off=0;
    while(!levels.empty()&&*levels.rbegin()>1){
        I top=*levels.rbegin();auto &bucket=buckets[top];int u=*bucket.begin();bucket.erase(u);if(bucket.empty())levels.erase(top);
        I remaining=d[u];d[u]=0;closed[u]=1;(*laid_off)++;
        std::vector<int>row;row.reserve(adj[u].size()+extra[u].size());
        for(int v:adj[u])if(!removed.count(key(u,v)))row.push_back(v);
        row.insert(row.end(),extra[u].begin(),extra[u].end());std::sort(row.begin(),row.end());row.erase(std::unique(row.begin(),row.end()),row.end());
        std::vector<int>selected;selected.reserve(remaining);
        // Skip empty degree levels; within each level preserve exact ID ordering.
        for(auto it=levels.rbegin();it!=levels.rend()&&remaining;it++){
            I level=*it;auto &pool=buckets[level];I take=std::min<I>(remaining,pool.size());
            if(take==I(pool.size()))selected.insert(selected.end(),pool.begin(),pool.end());
            else{
                std::vector<int>anchors;for(int v:row)if(d[v]==level&&v!=u){anchors.push_back(v);if(I(anchors.size())==take)break;}
                selected.insert(selected.end(),anchors.begin(),anchors.end());I left=take-anchors.size();
                if(left)for(int v:pool)if(!std::binary_search(anchors.begin(),anchors.end(),v)){selected.push_back(v);if(--left==0)break;}
            }
            remaining-=take;
        }
        if(remaining)throw std::runtime_error("partial realization failed");
        for(int v:selected){I old=d[v];buckets[old].erase(v);if(buckets[old].empty())levels.erase(old);d[v]=old-1;if(old>1){buckets[old-1].insert(v);levels.insert(old-1);}push(generated,u,v);}
    }
    std::copy(d.begin(),d.end(),residual);export_edges(generated,out);return I(generated.size());
 }catch(const std::exception&e){std::fprintf(stderr,"MCR ordered section: %s\n",e.what());return -1;}
}

extern "C" __declspec(dllexport) I matching_tail(int n,const I*active,I m,const I*virtual_edges,I*out){
 try{
    std::vector<bool>alive(n,false);std::set<int>remaining;for(int u=0;u<n;u++)if(active[u]){alive[u]=true;remaining.insert(u);}
    if(remaining.size()%2)throw std::runtime_error("odd matching boundary");
    std::vector<std::vector<int>>adj(n);
    for(I j=0;j<m;j++){int u=virtual_edges[2*j],v=virtual_edges[2*j+1];if(alive[u]&&alive[v]){adj[u].push_back(v);adj[v].push_back(u);}}
    for(auto&row:adj)std::sort(row.begin(),row.end());
    std::vector<std::pair<int,int>>edges;edges.reserve(remaining.size()/2);
    while(!remaining.empty()){
        int u=*remaining.begin();remaining.erase(u);alive[u]=false;int v=-1;
        for(int a:adj[u])if(alive[a]){v=a;break;}
        if(v<0)v=*remaining.begin();remaining.erase(v);alive[v]=false;push(edges,u,v);
    }
    export_edges(edges,out);return I(edges.size());
 }catch(const std::exception&e){std::fprintf(stderr,"MCR ordered tail: %s\n",e.what());return -1;}
}
