// Conditional-mixture centered moments for the frozen finite Gaussian chain.
// Forward log messages and exp weights keep the original binary64 operations.
// Central moments use extended precision and total-covariance/Welford merging.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

struct M {
    long double s=0, d=0, vs=0, vd=0, cov=0;
};
struct Mix {
    long double w=0, s=0, d=0, ms=0, md=0, cross=0;
    void add(double weight, const M& a, long double ds, long double dd) {
        if (weight == 0) return;
        const long double p=weight, ns=a.s+ds, nd=a.d+dd;
        const long double nw=w+p, delta_s=ns-s, delta_d=nd-d;
        const long double between=w*p/nw;
        ms += p*a.vs + between*delta_s*delta_s;
        md += p*a.vd + between*delta_d*delta_d;
        cross += p*a.cov + between*delta_s*delta_d;
        s += p/nw*delta_s;
        d += p/nw*delta_d;
        w=nw;
    }
    M finish() const {return M{s,d,ms/w,md/w,cross/w};}
};

EXPORT int route_s_centered_one_channel(
    std::int64_t count, const std::int64_t* offsets,
    const std::int64_t* states, const double* gain, const double* additive,
    const double* degree, const double* target, double inv, double coefficient,
    double residual_alpha, double* output, std::int64_t* counters) {
    if(count<=0 || !offsets || !states || !gain || !additive || !degree ||
       !target || !output || !counters || offsets[0]!=0 ||
       !std::isfinite(inv) || inv<=0 || !std::isfinite(coefficient) ||
       coefficient<=0 || !std::isfinite(residual_alpha)) return 1;
    std::int64_t width=0;
    for(std::int64_t i=0;i<count;++i){
        const auto k=offsets[i+1]-offsets[i]; if(k<=0) return 2;
        width=std::max(width,k);
    }
    std::vector<M> a(width),b(width);
    std::vector<double> fa(width),fb(width),weights(width),added(width);
    std::int64_t previous_width=offsets[1],work=0,underflow=0;
    const double first_offset=degree[0]-target[0];
    for(std::int64_t j=0;j<previous_width;++j){
        const double r=first_offset-static_cast<double>(states[j]);
        const double e=r*r*inv;
        fa[j]=gain[j]-r*r*coefficient;
        a[j].s=static_cast<long double>(additive[j])-static_cast<long double>(residual_alpha)*e;
        a[j].d=e;
    }
    for(std::int64_t i=1;i<count;++i){
        const auto po=offsets[i-1],co=offsets[i],cw=offsets[i+1]-co;
        if(previous_width>std::numeric_limits<std::int64_t>::max()/cw) return 3;
        work+=previous_width*cw;
        const double no=degree[i]-target[i];
        for(std::int64_t j=0;j<cw;++j){
            const double cs=static_cast<double>(states[co+j]);
            double maximum=-std::numeric_limits<double>::infinity();
            for(std::int64_t k=0;k<previous_width;++k){
                const double r=no+static_cast<double>(states[po+k])-cs;
                added[k]=r*r*inv;
                maximum=std::max(maximum,fa[k]-r*r*coefficient);
            }
            if(!std::isfinite(maximum)) return 4;
            double sum=0;
            for(std::int64_t k=0;k<previous_width;++k){
                const double r=no+static_cast<double>(states[po+k])-cs;
                const double value=fa[k]-r*r*coefficient;
                const double mass=std::exp(value-maximum);
                weights[k]=mass; sum+=mass; underflow+=(mass==0);
            }
            const double normalizer=maximum+std::log(sum);
            if(!std::isfinite(normalizer)) return 4;
            fb[j]=gain[co+j]+normalizer;
            Mix mix;
            for(std::int64_t k=0;k<previous_width;++k){
                weights[k]/=sum;
                const long double score=static_cast<long double>(additive[co+j])-
                    static_cast<long double>(residual_alpha)*added[k];
                mix.add(weights[k],a[k],score,added[k]);
            }
            b[j]=mix.finish();
        }
        a.swap(b);fa.swap(fb);previous_width=cw;
    }
    const auto lo=offsets[count-1],lw=offsets[count]-lo;
    const double no=degree[count]-target[count];
    double maximum=-std::numeric_limits<double>::infinity();
    for(std::int64_t j=0;j<lw;++j){
        const double r=no+static_cast<double>(states[lo+j]);
        added[j]=r*r*inv; maximum=std::max(maximum,fa[j]-r*r*coefficient);
    }
    if(!std::isfinite(maximum)) return 4;
    double sum=0;
    for(std::int64_t j=0;j<lw;++j){
        const double r=no+static_cast<double>(states[lo+j]);
        const double value=fa[j]-r*r*coefficient;
        weights[j]=std::exp(value-maximum);sum+=weights[j];
        underflow+=(weights[j]==0);
    }
    const double log_z=maximum+std::log(sum);
    if(!std::isfinite(log_z)) return 4;
    Mix mix;
    for(std::int64_t j=0;j<lw;++j){
        weights[j]/=sum;
        mix.add(weights[j],a[j],-static_cast<long double>(residual_alpha)*added[j],added[j]);
    }
    const M ans=mix.finish();
    output[0]=static_cast<double>(ans.s); output[1]=static_cast<double>(ans.d);
    output[2]=static_cast<double>(ans.vs); output[3]=static_cast<double>(ans.vd);
    output[4]=static_cast<double>(ans.cov); output[5]=log_z;
    counters[0]=work; counters[1]=underflow;
    counters[2]=std::numeric_limits<long double>::digits;
    if(!std::isfinite(ans.s)||!std::isfinite(ans.d)||!std::isfinite(ans.vs)||
       !std::isfinite(ans.vd)||!std::isfinite(ans.cov)||ans.vs<0||ans.vd<0) return 5;
    return 0;
}
