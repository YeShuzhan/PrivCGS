#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// The exact union of a small local frontier and the unchanged backing queue.
// A flush is a representation change only. It neither loses nor revisits a cell.
template<class T,class Lower,class Base> class ImmediateSuccessorQueue {
    static constexpr std::size_t CAPACITY=64;
    Base base_;
    Lower lower_;
    std::vector<T> local_;
    std::uint64_t pushed_=0,local_pops_=0,base_pops_=0,flushed_=0,partitions_=0;
    bool local_wins()const{return !local_.empty()&&(base_.empty()||!lower_(local_.front(),base_.top()));}
    void flush_lower_half(){
        const std::size_t keep=CAPACITY/2;
        auto better=[this](const T&a,const T&b){return lower_(b,a);};
        std::nth_element(local_.begin(),local_.begin()+keep,local_.end(),better);
        for(std::size_t i=keep;i<local_.size();++i){base_.push(local_[i]);++flushed_;}
        local_.resize(keep);
        std::make_heap(local_.begin(),local_.end(),lower_);++partitions_;
    }
public:
    ImmediateSuccessorQueue(){local_.reserve(CAPACITY);}
    bool empty()const{return local_.empty()&&base_.empty();}
    std::size_t size()const{return local_.size()+base_.size();}
    T top()const{if(empty())throw std::logic_error("empty local competition queue");return local_wins()?local_.front():base_.top();}
    void push(const T&v){++pushed_;local_.push_back(v);std::push_heap(local_.begin(),local_.end(),lower_);if(local_.size()==CAPACITY)flush_lower_half();}
    void pop(){if(local_wins()){std::pop_heap(local_.begin(),local_.end(),lower_);local_.pop_back();++local_pops_;}else{base_.pop();++base_pops_;}}
    ~ImmediateSuccessorQueue(){
        DiagnosticIO io;
        const char*p=std::getenv("S_SUCCESSOR_STATS");if(!p)return;
        if(auto f=std::fopen(p,"w")){
            std::fprintf(f,"{\"pushed\":%llu,\"bypassed_heap_pops\":%llu,\"base_pops\":%llu,\"flushed_to_base\":%llu,\"partitions\":%llu,\"local_capacity\":%llu,\"remaining\":%llu}\n",
                (unsigned long long)pushed_,(unsigned long long)local_pops_,(unsigned long long)base_pops_,(unsigned long long)flushed_,(unsigned long long)partitions_,(unsigned long long)CAPACITY,(unsigned long long)size());std::fclose(f);
        }
    }
};
