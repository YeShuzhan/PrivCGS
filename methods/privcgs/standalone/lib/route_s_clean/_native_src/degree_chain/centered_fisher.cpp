#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
// Conditional law of total covariance. No global raw second moments,
// eigenvalue clipping, changed weights, or PSD projection.
API int centered_fisher(int64_t n,int64_t k,const int64_t* off,const int64_t* st,
 const double* f,const double* gain,const double* a,const double* deg,const double* tgt,
 double iv,double coef,double* out,double* diag) {
 if(n<1||k<1||!(iv>0))return 1;
 const int64_t total=off[n],d=k+1;int64_t w=0;
 for(int64_t i=0;i<n;i++){if(off[i+1]<=off[i])return 2;w=std::max(w,off[i+1]-off[i]);}
 using L=long double;
 std::vector<L> mean(w*d),next(w*d),cov(w*d*d,0),nc(w*d*d),p(w),en(w),mix(d),delta(d);
 for(int64_t j=0;j<off[1];j++){
  for(int64_t c=0;c<k;c++)mean[j*d+c]=a[c*total+j];
  L r=L(deg[0])-tgt[0]-st[j];mean[j*d+k]=r*r*iv;
 }
 L maxerr=0;
 for(int64_t i=1;i<n;i++){
  const auto pw=off[i]-off[i-1],cw=off[i+1]-off[i];
  for(int64_t j=0;j<cw;j++){
   L mx=-std::numeric_limits<L>::infinity();
   for(int64_t b=0;b<pw;b++){L r=L(deg[i])-tgt[i]+st[off[i-1]+b]-st[off[i]+j];en[b]=r*r*iv;p[b]=L(f[off[i-1]+b])-coef*en[b];mx=std::max(mx,p[b]);}
   L sum=0;for(int64_t b=0;b<pw;b++){p[b]=std::exp(p[b]-mx);sum+=p[b];}
   if(!(sum>0)||!std::isfinite(sum))return 3;
   maxerr=std::max(maxerr,std::abs(mx+std::log(sum)+gain[off[i]+j]-f[off[i]+j]));
   std::fill(mix.begin(),mix.end(),0);
   for(int64_t b=0;b<pw;b++){p[b]/=sum;for(int64_t c=0;c<d;c++)mix[c]+=p[b]*(mean[b*d+c]+(c==k?en[b]:L(a[c*total+off[i]+j])));}
   for(int64_t c=0;c<d;c++)next[j*d+c]=mix[c];
   std::fill(nc.begin()+j*d*d,nc.begin()+(j+1)*d*d,0);
   for(int64_t b=0;b<pw;b++){
    for(int64_t c=0;c<d;c++)delta[c]=mean[b*d+c]+(c==k?en[b]:L(a[c*total+off[i]+j]))-mix[c];
    for(int64_t c=0;c<d;c++)for(int64_t e=0;e<d;e++)nc[j*d*d+c*d+e]+=p[b]*(cov[b*d*d+c*d+e]+delta[c]*delta[e]);
   }
  }
  mean.swap(next);cov.swap(nc);
 }
 auto lw=off[n]-off[n-1];L mx=-std::numeric_limits<L>::infinity();
 for(int64_t b=0;b<lw;b++){L r=L(deg[n])-tgt[n]+st[off[n-1]+b];en[b]=r*r*iv;p[b]=L(f[off[n-1]+b])-coef*en[b];mx=std::max(mx,p[b]);}
 L sum=0;for(int64_t b=0;b<lw;b++){p[b]=std::exp(p[b]-mx);sum+=p[b];}
 if(!(sum>0)||!std::isfinite(sum))return 4;
 std::fill(mix.begin(),mix.end(),0);
 for(int64_t b=0;b<lw;b++){p[b]/=sum;mean[b*d+k]+=en[b];for(int64_t c=0;c<d;c++)mix[c]+=p[b]*mean[b*d+c];}
 std::vector<L> result(d*d,0);
 for(int64_t b=0;b<lw;b++){for(int64_t c=0;c<d;c++)delta[c]=mean[b*d+c]-mix[c];for(int64_t c=0;c<d;c++)for(int64_t e=0;e<d;e++)result[c*d+e]+=p[b]*(cov[b*d*d+c*d+e]+delta[c]*delta[e]);}
 for(int64_t j=0;j<d*d;j++){out[j]=double(result[j]);if(!std::isfinite(out[j]))return 5;}
 diag[0]=double(mx+std::log(sum));diag[1]=double(maxerr);diag[2]=sizeof(L);return 0;
}
