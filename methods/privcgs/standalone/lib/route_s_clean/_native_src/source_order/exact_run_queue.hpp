#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Actual bound: O(buffer + read_block * live_runs) resident records.
// No quantization, altered comparator, delayed visibility, or sampling changes.
template<class T, class Lower> class ExactRunQueue {
    static_assert(std::is_trivially_copyable<T>::value, "disk record must be POD");
    static std::size_t option(const char* key, std::size_t fallback) {
        const char* p=std::getenv(key); return p?std::stoull(p):fallback;
    }
    struct Run {
        std::filesystem::path path;
        std::FILE* file=nullptr;
        std::vector<T> block;
        std::size_t pos=0, valid=0;
        std::uint64_t unread=0, file_bytes=0;
        ~Run(){if(file)std::fclose(file);}
        const T& head()const{return block[pos];}
        void refill(){
            valid=static_cast<std::size_t>(std::min<std::uint64_t>(block.size(),unread));pos=0;
            if(valid && std::fread(block.data(),sizeof(T),valid,file)!=valid)
                throw std::runtime_error("exact run queue truncated read");
            unread-=valid;
        }
    };
    struct HeadLower {
        const std::vector<std::unique_ptr<Run>>* runs; Lower lower;
        bool operator()(std::size_t a,std::size_t b)const{
            return lower((*runs)[a]->head(),(*runs)[b]->head());
        }
    };
    Lower lower_;
    bool shadow_on_=std::getenv("S_QUEUE_SHADOW")!=nullptr;
    std::priority_queue<T,std::vector<T>,Lower> shadow_;
    std::vector<T> ram_;
    std::vector<std::unique_ptr<Run>> runs_;
    std::priority_queue<std::size_t,std::vector<std::size_t>,HeadLower> heads_;
    std::filesystem::path dir_;
    std::size_t cap_,block_size_,max_runs_;
    std::uint64_t count_=0,bytes_=0,peak_bytes_=0,max_bytes_,min_free_,serial_=0;
    std::uint64_t written_=0,read_=0,flushes_=0,peak_runs_=0;
    bool ram_wins()const{
        return !ram_.empty() && (heads_.empty() ||
            !lower_(ram_.front(),runs_[heads_.top()]->head()));
    }
    static bool same_double(double a,double b){return std::memcmp(&a,&b,sizeof(double))==0;}
    static bool same_record(const T&a,const T&b){
        return a.historical.left==b.historical.left && a.historical.right==b.historical.right &&
            a.historical.path_hash==b.historical.path_hash && a.historical.serial==b.historical.serial &&
            same_double(a.historical.maximum,b.historical.maximum) &&
            same_double(a.historical.priority,b.historical.priority) && same_double(a.prune_priority,b.prune_priority);
    }
    void spill(){
        if(ram_.empty())return;
        if(heads_.size()>=max_runs_)throw std::runtime_error("exact run queue live-run budget exceeded");
        const auto size=ram_.size()*sizeof(T);
        if(size>max_bytes_-bytes_ || std::filesystem::space(dir_).available<size+min_free_)
            throw std::runtime_error("exact run queue scratch/free-space budget exceeded");
        // Reverse Lower orders highest first. Keys are unique among active cells.
        std::sort(ram_.begin(),ram_.end(),[&](const T&a,const T&b){return lower_(b,a);});
        auto r=std::make_unique<Run>();
        r->path=dir_/("run_"+std::to_string(serial_++)+".bin");
        r->file=std::fopen(r->path.string().c_str(),"w+b");
        if(!r->file)throw std::runtime_error("exact run queue cannot create owned run");
        if(std::fwrite(ram_.data(),sizeof(T),ram_.size(),r->file)!=ram_.size())
            throw std::runtime_error("exact run queue failed write");
        if(std::fflush(r->file)!=0 || std::fseek(r->file,0,SEEK_SET)!=0)
            throw std::runtime_error("exact run queue flush/seek failed");
        r->unread=ram_.size();r->file_bytes=size;r->block.resize(std::min(block_size_,ram_.size()));
        r->refill();read_+=r->valid*sizeof(T);
        bytes_+=size;peak_bytes_=std::max(peak_bytes_,bytes_);written_+=size;++flushes_;
        std::size_t slot=0;while(slot<runs_.size()&&runs_[slot])++slot;
        if(slot==runs_.size())runs_.push_back(nullptr);
        runs_[slot]=std::move(r);heads_.push(slot);ram_.clear();
        peak_runs_=std::max<std::uint64_t>(peak_runs_,heads_.size());
    }
public:
    ExactRunQueue():heads_(HeadLower{&runs_,Lower{}}),
        cap_(option("S_QUEUE_BUFFER_RECORDS",1u<<20)),
        block_size_(option("S_QUEUE_READ_RECORDS",2048)),
        max_runs_(option("S_QUEUE_MAX_RUNS",4096)),
        max_bytes_(option("S_QUEUE_MAX_BYTES",64ULL<<30)),
        min_free_(option("S_QUEUE_MIN_FREE_BYTES",16ULL<<30)){
        if(!cap_||!block_size_)throw std::runtime_error("queue capacities must be positive");
        const char* p=std::getenv("S_QUEUE_DIR");if(!p)throw std::runtime_error("S_QUEUE_DIR required");
        dir_=p;
        // Every experiment gets a fresh child directory; never delete a foreign path.
        if(!std::filesystem::create_directory(dir_))throw std::runtime_error("queue scratch directory must be new");
#ifdef _WIN32
        _setmaxstdio(8192);
#endif
        ram_.reserve(cap_);
    }
    ~ExactRunQueue(){
        const char* p=std::getenv("S_QUEUE_STATS");
        if(p){DiagnosticIO io;if(auto f=std::fopen(p,"w")){
            std::fprintf(f,"{\"buffer_records\":%llu,\"read_records\":%llu,\"flushes\":%llu,\"peak_runs\":%llu,\"peak_disk_bytes\":%llu,\"written_bytes\":%llu,\"read_bytes\":%llu,\"remaining_records\":%llu}\n",
                (unsigned long long)cap_,(unsigned long long)block_size_,(unsigned long long)flushes_,(unsigned long long)peak_runs_,(unsigned long long)peak_bytes_,(unsigned long long)written_,(unsigned long long)read_,(unsigned long long)count_);std::fclose(f);}}
        for(auto& r:runs_)if(r){auto path=r->path;r.reset();std::error_code ec;std::filesystem::remove(path,ec);}
        // Empty owned directory only (never recursive).
        std::error_code ec;std::filesystem::remove(dir_,ec);
    }
    ExactRunQueue(const ExactRunQueue&)=delete;
    bool empty()const{return count_==0;}
    std::size_t size()const{return static_cast<std::size_t>(count_);}
    const T& top()const{
        if(empty())throw std::logic_error("empty exact run queue top");
        const auto& result=ram_wins()?ram_.front():runs_[heads_.top()]->head();
        if(shadow_on_ && (shadow_.size()!=size() || !same_record(result,shadow_.top())))
            throw std::runtime_error("pathwise external/reference queue divergence");
        return result;
    }
    void push(const T&value){
        if(std::isnan(value.prune_priority)||std::isnan(value.historical.priority))
            throw std::runtime_error("NaN queue comparator input");
        ram_.push_back(value);std::push_heap(ram_.begin(),ram_.end(),lower_);++count_;
        if(shadow_on_)shadow_.push(value);
        if(ram_.size()==cap_)spill();
    }
    void pop(){
        if(empty())throw std::logic_error("empty exact run queue pop");
        if(shadow_on_){top();shadow_.pop();}
        if(ram_wins()){std::pop_heap(ram_.begin(),ram_.end(),lower_);ram_.pop_back();}
        else{
            auto i=heads_.top();heads_.pop();auto&r=*runs_[i];++r.pos;
            if(r.pos==r.valid && r.unread){r.refill();read_+=r.valid*sizeof(T);}
            if(r.pos<r.valid)heads_.push(i);
            else{auto path=r.path;bytes_-=r.file_bytes;runs_[i].reset();
                if(!std::filesystem::remove(path))throw std::runtime_error("failed to remove exhausted owned run");}
        }
        --count_;
    }
};
