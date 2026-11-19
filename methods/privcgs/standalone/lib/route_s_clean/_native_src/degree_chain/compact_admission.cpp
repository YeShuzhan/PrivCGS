#include "candidate_chart.cpp"
#include <limits>
#include <stdexcept>
namespace {
struct Keys {
 std::vector<std::uint64_t> table;std::size_t size=0;
 static constexpr std::uint64_t empty=UINT64_MAX;
 Keys():table(1024,empty){}
 static std::uint64_t hash(std::uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
 std::size_t slot(std::uint64_t x)const{auto i=hash(x)&(table.size()-1);while(table[i]!=empty&&table[i]!=x)i=(i+1)&(table.size()-1);return i;}
 bool has(std::uint64_t x)const{return table[slot(x)]==x;}
 void add(std::uint64_t x){
  if((size+1)*10>table.size()*7){auto old=std::move(table);table.assign(old.size()*2,empty);size=0;for(auto y:old)if(y!=empty){table[slot(y)]=y;++size;}}
  auto i=slot(x);if(table[i]==empty){table[i]=x;++size;}
 }
};
}
ROUTE_S_EXPORT int compact_admission(std::int64_t n,std::int64_t count,const double* gain,
 const std::int64_t* ol,const std::int64_t* or_,const std::int64_t* nl,const std::int64_t* nr,
 std::int64_t capacity,std::int64_t* selected,std::int64_t* out){
 if(n<=0||n>4294967295LL||count<0)return 1;
 try{
  std::vector<std::int64_t> order(count),scratch(count);std::iota(order.begin(),order.end(),std::int64_t{0});
  stable_radix_field(order,scratch,8,[&](std::int64_t i){return ~ordered_double_key(gain[i]);});
  std::vector<std::int64_t>().swap(scratch);
  Keys reserved;std::int64_t used=0;
  for(auto i:order){auto a=edge_key(n,ol[i],or_[i]),b=edge_key(n,nl[i],nr[i]);
   if(reserved.has(a)||reserved.has(b))continue;
   if(used==capacity)return 2;
   reserved.add(a);reserved.add(b);selected[used++]=i;
  }
  out[0]=used;out[1]=reserved.table.size()*sizeof(std::uint64_t);return 0;
 }catch(...){return 3;}
}
