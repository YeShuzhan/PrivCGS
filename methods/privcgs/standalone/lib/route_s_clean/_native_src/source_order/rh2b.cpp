#include <chrono>
#include <cstdio>
#include <stdexcept>
#include "diagnostic_timing.inc"
static long long observe_eta_calls=0,observe_eta_saturated=0;
struct ObserveSample { ObserveSample(int,bool){} };
#include "source_law.hpp"
#include <chrono>
#include <string>
#include <fstream>
#include <memory>

namespace rh2b {
using Clock=std::chrono::steady_clock;
static thread_local std::string error;
static thread_local double diagnostic_seconds=0;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double logadd(double a,double b){
    if(a==-INFINITY)return b;if(b==-INFINITY)return a;
    double hi=std::max(a,b),lo=std::min(a,b);
    return hi+std::log1p(std::exp(lo-hi));
}
struct LogTree {
    size_t base;std::vector<double> z;
    explicit LogTree(size_t n):base(1){while(base<n)base*=2;z.assign(2*base,-INFINITY);}
    void build(){for(size_t k=base-1;k;k--)z[k]=logadd(z[k*2],z[k*2+1]);}
    void update(size_t u,double x){size_t k=base+u;z[k]=x;while((k>>=1))z[k]=logadd(z[k*2],z[k*2+1]);}
    template<class R> int draw(R& rng)const{
        size_t k=1;
        while(k<base){double p=std::exp(z[k*2]-z[k]);k=k*2+(rng()>p);}
        return int(k-base);
    }
};
struct Random {
    uint64_t seed,domain,index=0;
    uint64_t integer_rejects=0;
    uint64_t word(){return splitmix64(seed ^ domain ^ (index++*GOLDEN64));}
    uint64_t below(uint64_t bound){
        if(!bound)throw std::runtime_error("empty uniform integer domain");
        uint64_t threshold=(uint64_t(0)-bound)%bound,x;
        do{x=word();if(x<threshold)integer_rejects++;}while(x<threshold);
        return x%bound;
    }
    double operator()(){
        // Open (0,1) with exactly representable 52-bit midpoints.
        uint64_t bits=word()>>12;
        return (double(bits)+0.5)*0x1p-52;
    }
};
struct EdgeSet {
    std::vector<uint64_t> keys;size_t mask;
    explicit EdgeSet(int64_t m){size_t cap=16;while(double(cap)*0.65<double(m))cap*=2;keys.assign(cap,UINT64_MAX);mask=cap-1;}
    bool contains(uint64_t key)const{size_t k=splitmix64(key)&mask;while(keys[k]!=UINT64_MAX){if(keys[k]==key)return true;k=(k+1)&mask;}return false;}
    void insert(uint64_t key){size_t k=splitmix64(key)&mask;while(keys[k]!=UINT64_MAX){if(keys[k]==key)throw std::runtime_error("duplicate insertion");k=(k+1)&mask;}keys[k]=key;}
};
uint64_t edgekey(int u,int v,int n){return uint64_t(std::min(u,v))*uint64_t(n)+uint64_t(std::max(u,v));}
double beta(const LawView& law,int v,int k){
    if(k==0||k==law.rank+2)return law.sociability[v];
    if(k<=law.rank)return law.sociability[v]*std::abs(law.interaction[int64_t(v)*law.rank+k-1]);
    if(k==law.rank+1)return law.star[v];return 1.0;
}
double alpha(const LawView& law,int u,int k){
    if(k==0)return (std::log(2.0)+1e-6)*law.sociability[u];
    if(k<=law.rank)return std::abs(law.weights[k-1])*law.sociability[u]*std::abs(law.interaction[int64_t(u)*law.rank+k-1]);
    if(k==law.rank+1)return law.sociability[u];
    if(k==law.rank+2)return law.star[u];return law.dust;
}
#include "ls2_support.hpp"
}
ROUTE_S_EXPORT const char* rh2b_error(){return rh2b::error.c_str();}
ROUTE_S_EXPORT double rh2b_diagnostic_io_seconds(){return rh2b::diagnostic_seconds;}
// Read-only primitive validation on supplied real graph snapshots; not a generator.
ROUTE_S_EXPORT int ls2_check_legal(int n,int64_t m,int64_t split,const int64_t* edges,
 int64_t q,const int64_t* queries,int64_t* output){
 try{
  rh2b::LegalIndex idx(n,m);std::vector<int> degree(n,0);
  for(int64_t i=0;i<m;i++){
   if(i==split)for(int64_t j=0;j<q;j++)output[j*2]=idx.select(int(queries[j*3]),uint64_t(queries[j*3+1]));
   int u=int(edges[2*i]),v=int(edges[2*i+1]);idx.add(u,v);degree[u]++;degree[v]++;
  }
  for(int64_t j=0;j<q;j++)output[j*2+1]=idx.select(int(queries[j*3]),uint64_t(queries[j*3+2]));
  rh2b::MissingTree tree(n,degree);
  for(int64_t u=0;u<n;u++)if(idx.root[u]>=0&&idx.size(idx.root[u])!=degree[u]+1)return 2;
  uint64_t prefix=0;for(int u=0;u<n;u++){uint64_t mass=n-1-degree[u];if(mass&&(tree.select(prefix)!=u||tree.select(prefix+mass-1)!=u))return 3;prefix+=mass;}
  return 0;
 }catch(...){return 1;}
}
ROUTE_S_EXPORT int rh2b_generate(
    int n,int rank,int parents,int components,int64_t m,double sigma,double tau,
    const double* target,const double* s,const double* star,const double* x,const double* weights,double dust,
    const int* parent,const int* component,const double* marked,const double* shift,const double* flux,
    uint64_t source_seed,uint64_t partner_seed,int64_t* output,int64_t* diag,double* real,double* masses,double* trace,
    const char* logpath,const char* stoppath
){
    using namespace rh2b;error.clear();diagnostic_seconds=0;
    auto start=Clock::now();int64_t admitted=0,rounds=0,invalid=0,eta_calls=0;
    int64_t duplicate_pair=0,max_retries=0,rescues=0,completion=0,exhausted=0,normal=0,saturated=0;
    double init=0;int status=0;long double sumc=0,sumc2=0,sumphi=0;double maxc=0;
    std::vector<uint64_t> histogram(65536,0);
    auto heartbeat=[&](const char* stage){
        auto begin=Clock::now();FILE* f=std::fopen(logpath,"ab");
        if(f){std::fprintf(f,"{\"stage\":\"%s\",\"native_seconds\":%.9g,\"admitted\":%lld,\"target_edges\":%lld,\"two_choice_rounds\":%lld,\"partner_draws\":%lld,\"invalid_candidates\":%lld,\"eta_calls\":%lld,\"frontier_records\":0}\n",stage,seconds(start),(long long)admitted,(long long)m,(long long)rounds,(long long)(2*rounds),(long long)invalid,(long long)eta_calls);std::fclose(f);}
        diagnostic_seconds+=seconds(begin);
    };
    try{
        if(n<2||rank<0||m<0||m>int64_t(n)*(n-1)/2||!(sigma>0)||!(tau>0)||!std::isfinite(sigma)||!std::isfinite(tau)||dust<0)throw std::runtime_error("invalid law domain");
        LawView law{n,rank,parents,components,s,star,x,weights,dust,parent,component,marked,shift,flux,0};
        const int K=rank+4;double invvar=1.0/(sigma*sigma);
        if(!(invvar>0)||!std::isfinite(invvar))throw std::runtime_error("invalid variance");
        std::vector<double> cdf(size_t(K)*(n+1)),logmass(n),coeff(K);
        std::vector<int> degree(n,0),support(K,0);LogTree tree(n),pressure(n);EdgeSet edges(m);
        LegalIndex legal(n,m);std::unique_ptr<MissingTree> missing;
        std::vector<long double> occupied_mass(n,0);
        heartbeat("carrier_build");
        for(int k=0;k<K;k++){
            double* c=&cdf[size_t(k)*(n+1)];c[0]=0;long double sum=0;
            for(int v=0;v<n;v++){double b=beta(law,v,k);if(!(b>=0)||!std::isfinite(b))throw std::runtime_error("nonnegative finite carrier violated");support[k]+=(b>0);sum+=b;c[v+1]=double(sum);}
            if(!std::isfinite(c[n]))throw std::runtime_error("carrier total overflow");
        }
        for(int u=0;u<n;u++){
            long double mass=0;
            for(int k=0;k<K;k++){
                const double* c=&cdf[size_t(k)*(n+1)];double a=alpha(law,u,k);
                if(!(a>=0)||!std::isfinite(a))throw std::runtime_error("invalid carrier coefficient");
                mass+=(long double)a*(c[u]+(c[n]-c[u+1]));
            }
            masses[u]=double(mass);
            if(masses[u]<0||!std::isfinite(masses[u])||!std::isfinite(target[u]))throw std::runtime_error("negative/nonfinite carrier row or target");
            logmass[u]=masses[u]>0?std::log(masses[u]):-INFINITY;tree.z[tree.base+u]=logmass[u]+target[u]*invvar;
            if(std::isnan(tree.z[tree.base+u])||tree.z[tree.base+u]==INFINITY)throw std::runtime_error("source log mass overflow");
        }
        tree.build();Random sr{source_seed,0x52483242534f5552ULL},pr{partner_seed,0x5248324250415254ULL},rr{partner_seed,0x4c53325245534355ULL},cr{source_seed,0x4c5332434f4d504cULL};
        auto illegal_fraction=[&](int u){
            if(!(masses[u]>0))return 0.0;
            double c=double(occupied_mass[u]/masses[u]);
            if(c<0||c>1){diag[19]++;real[13]=std::max(real[13],std::max(-c,c-1));}
            // Telemetry only. Never used to accept, reject, rescue, or draw sources.
            return std::max(0.,std::min(1.,c));
        };
        auto update=[&](int u){
            double lw=degree[u]==n-1?-INFINITY:logmass[u]+(target[u]-degree[u])*invvar;
            tree.update(u,lw);double c=illegal_fraction(u);pressure.update(u,c>0&&lw!=-INFINITY?lw+2*std::log(c):-INFINITY);
        };
        init=seconds(start)-diagnostic_seconds;heartbeat("birth");auto last=Clock::now();
        while(admitted<m){
            bool complete=(tree.z[1]==-INFINITY);
            if(complete&&!missing)missing.reset(new MissingTree(n,degree));
            int u=complete?missing->select(cr.below(missing->total)):tree.draw(sr);
            if(u<0||u>=n)throw std::runtime_error("invalid source draw");
            if(degree[u]==n-1)throw std::runtime_error("saturated source must have zero source mass");
            double c=illegal_fraction(u),phi=complete?0.0:(pressure.z[1]==-INFINITY?0.0:std::exp(pressure.z[1]-tree.z[1]));
            if(!complete){normal++;sumc+=c;sumc2+=c*c;sumphi+=phi;maxc=std::max(maxc,c);histogram[std::min(65535,int(c*65536))]++;}
            double cmass=0;
            for(int k=0;k<K;k++){const double* c=&cdf[size_t(k)*(n+1)];cmass+=alpha(law,u,k)*(c[u]+(c[n]-c[u+1]));coeff[k]=cmass;}
            auto partner_draw=[&](){
                double t=pr()*cmass;int k=int(std::upper_bound(coeff.begin(),coeff.end(),t)-coeff.begin());if(k==K)k=K-1;
                const double* c=&cdf[size_t(k)*(n+1)];double left=c[u],right=c[n]-c[u+1],point=pr()*(left+right);
                int v;
                if(point<left)v=int(std::upper_bound(c+1,c+u+1,point)-c)-1;
                else {point=c[u+1]+(point-left);if(point>=c[n])point=std::nextafter(c[n],-INFINITY);v=int(std::upper_bound(c+u+2,c+n+1,point)-c)-1;}
                if(v<0||v>=n||v==u)throw std::runtime_error("conditional carrier CDF numeric failure");return v;
            };
            int v1=-1,v2=-1,v=-1,x1=-1,x2=-1,branch=0;uint64_t k1=0,k2=0;double j1=-INFINITY,j2=-INFINITY;
            if(complete){
                branch=2;completion++;k1=cr.below(uint64_t(n-1-degree[u]));v=legal.select(u,k1);v1=v2=v;
            }else{
                v1=partner_draw();v2=partner_draw();x1=v1;x2=v2;rounds++;
                bool legal1=!edges.contains(edgekey(u,v1,n)),legal2=!edges.contains(edgekey(u,v2,n));
                invalid+=(!legal1)+(!legal2);duplicate_pair+=(v1==v2);
                if(!legal1&&!legal2){
                    branch=1;rescues++;legal.activate(u);
                    bool covered=true;
                    for(int k=0;k<K&&covered;k++)if(alpha(law,u,k)>0){
                        int covered_count=(beta(law,u,k)>0);
                        for(auto i=legal.head[u];i>=0;i=legal.next[i])covered_count+=(beta(law,legal.neighbor[i],k)>0);
                        if(covered_count<support[k])covered=false;
                    }
                    exhausted+=covered;
                    k1=rr.below(uint64_t(n-1-degree[u]));k2=rr.below(uint64_t(n-1-degree[u]));v1=legal.select(u,k1);v2=legal.select(u,k2);
                    legal1=legal2=true;
                }
                j1=legal1?edge_eta(law,u,v1)/tau+(target[v1]-degree[v1])*invvar:-INFINITY;
                j2=legal2?edge_eta(law,u,v2)/tau+(target[v2]-degree[v2])*invvar:-INFINITY;
                eta_calls+=int(legal1)+int(legal2);
                if((legal1&&!std::isfinite(j1))||(legal2&&!std::isfinite(j2)))throw std::runtime_error("nonfinite exact eta score");
                if(legal1||legal2)v=(j1>j2||(j1==j2&&v1<v2))?v1:v2;
            }
            if(v<0||v==u||edges.contains(edgekey(u,v,n)))throw std::runtime_error("LS2 birth legality invariant");
            max_retries=std::max(max_retries,int64_t(!complete));
            if(admitted<128){double* t=trace+admitted*18;t[0]=u;t[1]=v1;t[2]=v2;t[3]=v;t[4]=j1;t[5]=j2;t[6]=degree[u];t[7]=degree[v1];t[8]=degree[v2];t[9]=tree.z[tree.base+u];t[10]=tree.z[1];t[11]=!complete;t[12]=branch;t[13]=c;t[14]=x1;t[15]=x2;t[16]=phi;t[17]=n-1-degree[u];}
            if(branch){auto begin=Clock::now();std::ofstream f(std::string(logpath)+".interventions",std::ios::app);f.precision(17);f<<"{\"birth\":"<<admitted<<",\"branch\":"<<branch<<",\"u\":"<<u<<",\"x1\":"<<x1<<",\"x2\":"<<x2<<",\"y1\":"<<v1<<",\"y2\":"<<v2<<",\"k1\":"<<k1<<",\"k2\":"<<k2<<",\"winner\":"<<v<<",\"d_u\":"<<degree[u]<<",\"target_u\":"<<target[u]<<",\"M_u\":"<<masses[u]<<",\"c_u\":"<<c<<",\"Phi_t\":"<<phi<<"}\n";f.close();diagnostic_seconds+=seconds(begin);}
            edges.insert(edgekey(u,v,n));output[2*admitted]=std::min(u,v);output[2*admitted+1]=std::max(u,v);admitted++;
            legal.add(u,v);
            long double uv=0,vu=0;for(int k=0;k<K;k++){uv+=(long double)alpha(law,u,k)*beta(law,v,k);vu+=(long double)alpha(law,v,k)*beta(law,u,k);}
            occupied_mass[u]+=uv;occupied_mass[v]+=vu;
            degree[u]++;degree[v]++;
            saturated+=(degree[u]==n-1)+(degree[v]==n-1);update(u);update(v);if(missing){missing->remove_one(u);missing->remove_one(v);}
            if((admitted&65535)==0&&seconds(last)>5){heartbeat("birth");last=Clock::now();std::ifstream stop(stoppath);if(stop.good())throw std::runtime_error("user/resource stop requested");}
        }
        long double debt=0;int over=0;for(int u=0;u<n;u++){debt+=std::abs(degree[u]-target[u]);over+=(degree[u]>target[u]);}
        real[2]=double(debt);diag[6]=over;diag[7]=int64_t(edges.keys.size()*sizeof(uint64_t));diag[8]=int64_t(cdf.size()*sizeof(double));diag[9]=int64_t(tree.z.size()*sizeof(double));
        diag[10]=rescues;diag[11]=completion;diag[12]=exhausted;diag[13]=saturated;diag[14]=legal.bytes();diag[15]=legal.activations;diag[16]=legal.rank_queries;diag[17]=rr.integer_rejects+cr.integer_rejects;diag[18]=normal;
        real[4]=normal?double(sumc/normal):0;real[5]=normal?double(sumc2/normal):0;real[6]=normal?double(sumphi/normal):0;real[7]=maxc;
        double percentiles[]={.95,.99,.999};for(int j=0;j<3;j++){uint64_t wanted=uint64_t(std::ceil(percentiles[j]*normal)),running=0;int b=0;for(;b<65535;b++){running+=histogram[b];if(running>=wanted)break;}real[8+j]=double(b)/65536;}
        real[11]=normal?double(rescues)/normal:0;real[12]=double(pressure.z.size()*sizeof(double)+occupied_mass.size()*sizeof(long double));
        heartbeat("complete");
    }catch(const std::exception& exc){error=exc.what();status=1;heartbeat("failed");}
    diag[0]=admitted;diag[1]=rounds;diag[2]=invalid;diag[3]=eta_calls;diag[4]=duplicate_pair;diag[5]=max_retries;
    real[0]=seconds(start)-diagnostic_seconds;real[1]=init;real[3]=diagnostic_seconds;
    return status;
}
