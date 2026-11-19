// Equivalent frozen W4 proposal order and JR integer-risk acceptance.
// All graph arithmetic is integral. No random draw, floating score, or fast-math.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
using I=long long; using U=uint64_t; using V=uint32_t;
static thread_local std::string error;
static U mix(U x,U seed=0){x+=seed+0x9E3779B97F4A7C15ULL;x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;x=(x^(x>>27))*0x94D049BB133111EBULL;return x^(x>>31);}
static U key(V a,V b){if(a>b)std::swap(a,b);return (U(a)<<32)|b;}
// Fixed-load open addressing, backward-shift deletion: no accumulating tombstones.
struct EdgeMap {
 std::vector<U> k;std::vector<size_t> ids;size_t mask;
 explicit EdgeMap(size_t m){size_t c=16;while(c<m*2+1)c*=2;k.assign(c,0);ids.resize(c);mask=c-1;}
 bool has(U x)const{size_t p=mix(x)&mask;while(k[p]){if(k[p]==x)return true;p=(p+1)&mask;}return false;}
 size_t index(U x)const{size_t p=mix(x)&mask;while(k[p]){if(k[p]==x)return ids[p];p=(p+1)&mask;}throw std::runtime_error("edge index missing");}
 void set_index(U x,size_t id){size_t p=mix(x)&mask;while(k[p]){if(k[p]==x){ids[p]=id;return;}p=(p+1)&mask;}throw std::runtime_error("edge index missing");}
 void add(U x,size_t id){if(!x)throw std::runtime_error("zero edge key");size_t p=mix(x)&mask;while(k[p]){if(k[p]==x)throw std::runtime_error("duplicate edge");p=(p+1)&mask;}k[p]=x;ids[p]=id;}
 void erase(U x){size_t p=mix(x)&mask;while(k[p]&&k[p]!=x)p=(p+1)&mask;if(!k[p])throw std::runtime_error("missing removed edge");size_t hole=p;
  for(p=(p+1)&mask;k[p];p=(p+1)&mask){size_t h=mix(k[p])&mask;if(((hole-h)&mask)<((p-h)&mask)){k[hole]=k[p];ids[hole]=ids[p];hole=p;}}k[hole]=0;}
};
// Arbitrary precision nonnegative integers, little-endian 64-bit limbs.
// Needed because frozen exact-risk slopes can exceed machine integer ranges.
struct Nat {
 std::vector<U> x;
 Nat(U a=0){if(a)x.push_back(a);}
 void trim(){while(!x.empty()&&!x.back())x.pop_back();}
 int cmp(const Nat& b)const{if(x.size()!=b.x.size())return x.size()<b.x.size()?-1:1;for(size_t i=x.size();i-->0;)if(x[i]!=b.x[i])return x[i]<b.x[i]?-1:1;return 0;}
 Nat mul(U a)const{Nat o;if(!a)return o;o.x.resize(x.size());U c=0;for(size_t i=0;i<x.size();++i){unsigned __int128 v=(unsigned __int128)x[i]*a+c;o.x[i]=(U)v;c=(U)(v>>64);}if(c)o.x.push_back(c);return o;}
 void add(U a){size_t i=0;while(a){if(i==x.size())x.push_back(0);U old=x[i];x[i]+=a;a=x[i]<old;++i;}}
 static Nat parse(const char* s){Nat o;for(;*s;++s){if(*s<'0'||*s>'9')throw std::runtime_error("invalid exact risk integer");o=o.mul(10);o.add(*s-'0');}return o;}
 Nat sub(const Nat& b)const{if(cmp(b)<0)throw std::runtime_error("negative natural subtraction");Nat o;o.x.resize(x.size());U carry=0;for(size_t i=0;i<x.size();++i){U q=i<b.x.size()?b.x[i]:0;U v=x[i]-q;U c1=x[i]<q;U w=v-carry;U c2=v<carry;o.x[i]=w;carry=c1|c2;}if(carry)throw std::runtime_error("integer borrow");o.trim();return o;}
};
struct Signed {
 int sign;Nat mag;
 Signed(const Nat& a,const Nat& b){int c=a.cmp(b);sign=c;mag=c>=0?a.sub(b):b.sub(a);}
 int cmp(const Signed& b)const{if(sign!=b.sign)return sign<b.sign?-1:1;return sign<0?-mag.cmp(b.mag):mag.cmp(b.mag);}
};
struct Stats {I attempted=0,accepted=0,gain=0,open_wedges=0,scanned=0,nonpositive=0,overshoot=0,kinds[5]={0,0,0,0,0},valid=0,reject_hard=0,reject_risk=0,cross=0,current=0;double seconds=0;};
using Progress=void(*)(I,I,I,const I*);
struct Policy {
 bool risk;I lo,hi,bandlo,bandhi,target;Nat a,b,den;Signed current_num;
 Policy(bool r,I l,I h,I bl,I bh,I t,const char* aa,const char* bb,const char* dd,I current):risk(r),lo(l),hi(h),bandlo(bl),bandhi(bh),target(t),a(Nat::parse(aa)),b(Nat::parse(bb)),den(Nat::parse(dd)),current_num(number(current)){}
 Signed number(I t)const{if(t<0)throw std::runtime_error("negative triangle count");Signed l(a.mul((U)t),den),r(den,b.mul((U)t));return l.cmp(r)>=0?l:r;}
 bool hit(I t)const{return risk?bandlo<=t&&t<=bandhi:lo<=t&&t<=hi;}
 bool allow(I old,I next,int direction,Stats& s){
  if(risk){++s.valid;if(next<lo||next>hi){++s.reject_hard;return false;}if(next==old||number(next).cmp(current_num)>=0){++s.reject_risk;return false;}return true;}
  I delta=next-old;if((direction>0&&delta<=0)||(direction<0&&delta>=0)){++s.nonpositive;return false;}
  if((direction>0&&next>hi)||(direction<0&&next<lo)){++s.overshoot;return false;}return true;
 }
 void committed(I old,I next,Stats& s){if(risk){if((old<target&&next>target)||(old>target&&next<target))++s.cross;current_num=number(next);}}
};
struct Graph {
 I* edges;V n;size_t m;int threads;EdgeMap map;
 std::vector<size_t> off;std::vector<V> deg,len,neighbor;std::vector<size_t> incident,pos0,pos1;
 std::vector<U> version,ckey,cva,cvb;std::vector<I> supports;U epoch=0,full_epoch=~U(0);I full_tri=0;
 std::vector<I> ledger;
 Graph(I* e,V nn,size_t mm,int th):edges(e),n(nn),m(mm),threads(th),map(mm),off(nn+1,0),deg(nn,0),len(nn,0),pos0(mm),pos1(mm),version(nn,0),ckey(mm,0),cva(mm,0),cvb(mm,0),supports(mm,0){
  if(nn>=0x80000000U)throw std::runtime_error("vertex domain too large");
  for(size_t i=0;i<m;++i){I a=e[2*i],b=e[2*i+1];if(a<0||a>=b||b>=nn)throw std::runtime_error("noncanonical edge");++deg[a];++deg[b];map.add(key((V)a,(V)b),i);}
  unsigned __int128 W=0;for(V u=0;u<n;++u){off[u+1]=off[u]+deg[u];W+=(unsigned __int128)deg[u]*(deg[u]-I(1))/2;}
  if(W>(U)std::numeric_limits<I>::max())throw std::runtime_error("original int64 wedge domain exceeded");
  neighbor.resize(2*m);incident.resize(2*m);reindex();
 }
 void reindex(){std::fill(len.begin(),len.end(),0);for(size_t i=0;i<m;++i){V a=(V)edges[2*i],b=(V)edges[2*i+1];size_t p=off[a]+len[a]++,q=off[b]+len[b]++;neighbor[p]=b;neighbor[q]=a;incident[p]=incident[q]=i;pos0[i]=p;pos1[i]=q;}if(len!=deg)throw std::runtime_error("degree state changed");}
 void bind(I* e,size_t mm){if(mm!=m)throw std::runtime_error("graph size changed at bind");triangles();std::vector<I> remapped(m);for(size_t i=0;i<m;++i){I a=e[2*i],b=e[2*i+1];if(a<0||a>=b||b>=n||!map.has(key((V)a,(V)b)))throw std::runtime_error("graph state differs at bind");if(i&&std::tie(e[2*i-2],e[2*i-1])>=std::tie(e[2*i],e[2*i+1]))throw std::runtime_error("bind input not uniquely canonical");remapped[i]=supports[map.index(key((V)a,(V)b))];}
  edges=e;reindex();supports.swap(remapped);for(size_t i=0;i<m;++i){V a=(V)e[2*i],b=(V)e[2*i+1];U k=key(a,b);map.set_index(k,i);ckey[i]=k;cva[i]=version[a];cvb[i]=version[b];} // Same topology and exact verified counts, newly canonical edge indices.
 }
 bool has(V a,V b)const{return a!=b&&map.has(key(a,b));}
 I common(V a,V b)const{if(len[a]>len[b])std::swap(a,b);I c=0;for(size_t p=off[a];p<off[a]+len[a];++p)c+=has(neighbor[p],b);return c;}
 I support(size_t i){V a=(V)edges[2*i],b=(V)edges[2*i+1];U k=key(a,b);if(ckey[i]==k&&cva[i]==version[a]&&cvb[i]==version[b])return supports[i];I s=common(a,b);ckey[i]=k;cva[i]=version[a];cvb[i]=version[b];supports[i]=s;return s;}
 I triangles(){if(full_epoch==epoch)return full_tri;I total=0;
  #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:total)
  for(I i=0;i<(I)m;++i)total+=support((size_t)i);
  if(total%3)throw std::runtime_error("triangle incidence remainder");full_tri=total/3;full_epoch=epoch;return full_tri;}
 V other(size_t i,V u)const{V a=(V)edges[2*i],b=(V)edges[2*i+1];return a==u?b:b==u?a:n;}
 void remove_neighbor(V a,size_t edgeid){size_t p=(V)edges[2*edgeid]==a?pos0[edgeid]:pos1[edgeid];size_t last=off[a]+--len[a];if(p!=last){neighbor[p]=neighbor[last];size_t j=incident[last];incident[p]=j;if((V)edges[2*j]==a)pos0[j]=p;else pos1[j]=p;}}
 void add_neighbor(V a,V b,size_t edgeid){size_t p=off[a]+len[a]++;if(len[a]>deg[a])throw std::runtime_error("neighbor capacity");neighbor[p]=b;incident[p]=edgeid;if((V)edges[2*edgeid]==a)pos0[edgeid]=p;else pos1[edgeid]=p;}
 I delta(size_t first,size_t second,V a,V u,V b,V v,V x,V y,V z,V w,bool& valid,I known_first_common=-1){
  valid=false;if(first==second||a==u||a==b||a==v||u==b||u==v||b==v)return 0;
  if(other(first,a)!=u||other(second,b)!=v||x==y||z==w||key(x,y)==key(z,w)||has(x,y)||has(z,w))return 0;
  std::array<V,4> ids={a,u,b,v};auto partner=[&](V t){if(t==a)return u;if(t==u)return a;if(t==b)return v;if(t==v)return b;throw std::runtime_error("nonmatching swap");};
  auto new_support=[&](V s,V t){return (s==x&&t==y&&known_first_common>=0?known_first_common:common(s,t))-I(has(partner(s),t))-I(has(partner(t),s));};
  I d=-support(first)-support(second)+new_support(x,y)+new_support(z,w);valid=true;return d;
 }
 void commit(size_t first,size_t second,V a,V u,V b,V v,V x,V y,V z,V w,I d,std::ofstream* trace,bool saveledger){
  remove_neighbor(a,first);remove_neighbor(u,first);remove_neighbor(b,second);remove_neighbor(v,second);map.erase(key(a,u));map.erase(key(b,v));
  edges[2*first]=std::min(x,y);edges[2*first+1]=std::max(x,y);edges[2*second]=std::min(z,w);edges[2*second+1]=std::max(z,w);map.add(key(x,y),first);map.add(key(z,w),second);
  add_neighbor(x,y,first);add_neighbor(y,x,first);add_neighbor(z,w,second);add_neighbor(w,z,second);++version[a];++version[u];++version[b];++version[v];++epoch;
  I record[9]={a,u,b,v,x,y,z,w,d};if(saveledger)ledger.insert(ledger.end(),record,record+9);if(trace&&*trace)trace->write(reinterpret_cast<char*>(record),sizeof(record));
 }
 Stats increase(I current,I low,I high,I sweep,Policy& policy,Progress progress,std::ofstream* trace){
  auto start=std::chrono::steady_clock::now();Stats s;triangles();std::vector<std::array<I,4>> donor(n);
  #pragma omp parallel for num_threads(threads) schedule(static)
  for(I node=0;node<(I)n;++node){std::array<std::pair<I,I>,4> best;for(auto& b:best)b={std::numeric_limits<I>::max(),-1};
   for(size_t p=off[node];p<off[node]+len[node];++p){size_t i=incident[p];V othernode=neighbor[p];I score=(len[othernode]==1?0:len[othernode]==2?1:2)*I(n+1)+supports[i];std::pair<I,I> v={score,(I)i};if(v>=best[3])continue;int j=3;while(j&&v<best[j-1]){best[j]=best[j-1];--j;}best[j]=v;}
   for(int j=0;j<4;++j)donor[node][j]=best[j].second;}
  U seed=(1517938221ULL+U(sweep)*0x9E3779B1ULL)&~U(0);std::vector<std::pair<U,V>> pivot_keys;for(V u=0;u<n;++u)if(deg[u]>=2)pivot_keys.push_back({mix(u,seed^0xA24BAED4963EE407ULL),u});
  std::sort(pivot_keys.begin(),pivot_keys.end());
  struct C {I low,leaf,support;size_t first,second;V u,v,du,dv;};
  std::vector<std::pair<U,V>> neigh;
  for(size_t position=0;position<pivot_keys.size();++position){if(policy.hit(current))break;V pivot=pivot_keys[position].second;U ns=seed^(U(pivot+1)*0xD6E8FEB86659FD93ULL);neigh.clear();neigh.reserve(len[pivot]);for(size_t p=off[pivot];p<off[pivot]+len[pivot];++p)neigh.push_back({mix(neighbor[p],ns),neighbor[p]});
   std::sort(neigh.begin(),neigh.end());
   for(size_t cursor=0;cursor+1<neigh.size();cursor+=2){if(policy.hit(current))break;V a=neigh[cursor].second,b=neigh[cursor+1].second;if(has(a,b))continue;++s.open_wedges;std::array<C,16> cs;size_t count=0;
    for(I fi:donor[a]){if(fi<0)continue;size_t first=(size_t)fi;V u=other(first,a);if(u==n||u==pivot||u==b)continue;
     for(I se:donor[b]){++s.scanned;if(se<0||(size_t)se==first)continue;size_t second=(size_t)se;V v=other(second,b);if(v==n||v==pivot||a==u||a==b||a==v||u==b||u==v||b==v||has(u,v))continue;
      V du=deg[u],dv=deg[v];cs[count++]={-I((du<=2)+(dv<=2)),-I((du==1)+(dv==1)),support(first)+support(second),first,second,u,v,du,dv};}}
    std::sort(cs.begin(),cs.begin()+count,[](const C& a,const C& b){return std::tie(a.low,a.leaf,a.support,a.first,a.second)<std::tie(b.low,b.leaf,b.support,b.first,b.second);});
    I new_ab=count?common(a,b):-1;
    for(size_t ci=0;ci<count;++ci){const C& c=cs[ci];++s.attempted;bool valid;I d=delta(c.first,c.second,a,c.u,b,c.v,a,b,c.u,c.v,valid,new_ab);if(!valid)continue;if(!policy.allow(current,current+d,1,s))continue;
     I old=current;commit(c.first,c.second,a,c.u,b,c.v,a,b,c.u,c.v,d,trace,false);current+=d;s.gain+=d;++s.accepted;policy.committed(old,current,s);V dl=std::min(c.du,c.dv),dh=std::max(c.du,c.dv);int type=dl==1&&dh==1?0:dl==1&&dh==2?1:dl==2&&dh==2?2:dl<=2?3:4;++s.kinds[type];if(type==0&&(len[c.u]!=1||neighbor[off[c.u]]!=c.v||len[c.v]!=1||neighbor[off[c.v]]!=c.u))throw std::runtime_error("degree1 pair");break;}
   }
   if(position&&position%10000==0&&progress)progress((I)position,s.accepted,current,s.kinds);
  }
  if(len!=deg)throw std::runtime_error("degree invariant");s.current=current;s.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return s;
 }
 Stats decrease(I current,I low,I high,I sweep,Policy& policy,std::ofstream* trace){
  auto start=std::chrono::steady_clock::now();Stats s;ledger.clear();triangles();std::vector<size_t> order(m);
  // Exact stable counting order equals original lexsort((edge_id,support)).
  I maximum=0;for(I v:supports)maximum=std::max(maximum,v);std::vector<size_t> starts((size_t)maximum+1,0);
  for(I v:supports)++starts[(size_t)v];size_t total=0;for(size_t& v:starts){size_t count=v;v=total;total+=count;}
  for(size_t i=0;i<m;++i)order[starts[(size_t)supports[i]]++]=i;
  for(size_t pos=0;pos<m;++pos){if(policy.hit(current))break;size_t first=order[m-1-pos];V a=(V)edges[2*first],b=(V)edges[2*first+1];if(!support(first))continue;bool success=false;
   for(size_t slot=0;slot<std::min(size_t(4),m);++slot){size_t second=order[(pos*4+slot+U(sweep))%m];V c=(V)edges[2*second],d=(V)edges[2*second+1];if(a==b||a==c||a==d||b==c||b==d||c==d)continue;
    std::array<std::array<V,4>,2> added={{{a,c,b,d},{a,d,b,c}}};
    for(auto p:added){++s.attempted;bool valid;I deltaT=delta(first,second,a,b,c,d,p[0],p[1],p[2],p[3],valid);if(!valid)continue;if(!policy.allow(current,current+deltaT,-1,s))continue;
     I old=current;commit(first,second,a,b,c,d,p[0],p[1],p[2],p[3],deltaT,trace,true);current+=deltaT;s.gain+=deltaT;++s.accepted;policy.committed(old,current,s);success=true;break;}
    if(success)break;}
  }
  if(len!=deg)throw std::runtime_error("degree invariant");s.current=current;s.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return s;
 }
};
API const char* eq_error(){return error.c_str();}
API void* eq_create(I* edges,I n,I m,int threads){try{if(n<0||n>=0x80000000LL||m<0||threads<1)throw std::runtime_error("invalid dimensions");return new Graph(edges,(V)n,(size_t)m,threads);}catch(const std::exception& e){error=e.what();return nullptr;}}
API void eq_free(void* p){delete static_cast<Graph*>(p);}
API int eq_bind(void* p,I* edges,I m){try{static_cast<Graph*>(p)->bind(edges,(size_t)m);return 0;}catch(const std::exception& e){error=e.what();return -1;}}
API I eq_triangles(void* p){try{return static_cast<Graph*>(p)->triangles();}catch(const std::exception& e){error=e.what();return -1;}}
API I eq_common(void* p,I a,I b){try{return static_cast<Graph*>(p)->common((V)a,(V)b);}catch(const std::exception& e){error=e.what();return -1;}}
API int eq_sweep(void* p,int direction,I current,I low,I high,I sweep,int risk,I bl,I bh,I target,const char* a,const char* b,const char* den,Stats* result,Progress progress,const char* tracepath){
 try{Graph& g=*static_cast<Graph*>(p);Policy policy(risk,low,high,bl,bh,target,a,b,den,current);std::ofstream trace;if(tracepath&&*tracepath){trace.open(tracepath,std::ios::binary|std::ios::app);if(!trace)throw std::runtime_error("trace open");}*result=direction>0?g.increase(current,low,high,sweep,policy,progress,&trace):g.decrease(current,low,high,sweep,policy,&trace);return 0;}catch(const std::exception& e){error=e.what();return -1;}}
API I eq_ledger_count(void* p){return (I)static_cast<Graph*>(p)->ledger.size()/9;}
API const I* eq_ledger_data(void* p){return static_cast<Graph*>(p)->ledger.data();}
