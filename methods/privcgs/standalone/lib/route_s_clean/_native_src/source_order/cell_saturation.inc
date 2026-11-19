thread_local bool bound_cert_enabled=false,bound_cert_shadow=false;
thread_local double bound_cert_cap=0.0;
thread_local std::uint64_t bound_cert_trials=0,bound_cert_success=0;

void initialize_bound_certificate(const LawView&law,const TreeView&tree){
    bound_cert_enabled=false;bound_cert_shadow=std::getenv("S_BOUND_SHADOW")!=nullptr;
    bound_cert_trials=bound_cert_success=0;
    const double limit=std::ldexp(1.0,100);
    if(law.rank<1||law.rank>64||tree.rank!=law.rank||!std::isfinite(law.dust)||law.dust<0||law.dust>limit)return;
    for(int k=0;k<law.rank;++k)if(!std::isfinite(law.weights[k])||std::abs(law.weights[k])>limit)return;
    for(int u=0;u<tree.count;++u){
        const double s=tree.sociability_max[u],t=tree.star_max[u];
        if(!std::isfinite(s)||!std::isfinite(t)||s<0||t<0||s>limit||t>limit)return;
        const auto offset=static_cast<std::int64_t>(u)*law.rank;
        for(int k=0;k<law.rank;++k){
            const double lo=tree.low[offset+k],hi=tree.high[offset+k];
            if(!std::isfinite(lo)||!std::isfinite(hi)||std::abs(lo)>limit||std::abs(hi)>limit)return;
        }
    }
    volatile double cap=20.0;bound_cert_cap=log_expm1_intensity(cap);
    bound_cert_enabled=true;
}

CellBoundsOperatorBound cell_bounds_operator_bound(
    const LawView&law,const TreeView&tree,const JointTreeSummaryOperatorBound&summary,int left,int right
){
    if(singleton(tree,left)&&singleton(tree,right))
        return cell_bounds_operator_bound_original(law,tree,summary,left,right);
    ++last_operator_bound_joint_bound_call_count;
    ++bound_cert_trials;
    if(bound_cert_enabled){
        const double floor=(0.0+tree.sociability_max[left]*tree.star_max[right])+
            tree.sociability_max[right]*tree.star_max[left]+law.dust;
        if(floor>=20.0){
            const double frozen_parent=cached_marked_upper(law,tree,left)+
                cached_marked_upper(law,tree,right)+
                cached_parent_upper(law,tree,left,right);
            const double value=bound_cert_cap+frozen_parent+law.component_upper+1.0e-12;
            const CellBoundsOperatorBound result{value,value};
            ++bound_cert_success;
            if(bound_cert_shadow){
                const auto original=cell_bounds_operator_bound_original(law,tree,summary,left,right);
                if(std::memcmp(&result.frozen,&original.frozen,sizeof(double)) ||
                   std::memcmp(&result.joint,&original.joint,sizeof(double)))
                    throw std::runtime_error("cell saturation certificate disagrees with original binary64 bounds");
            }
            return result;
        }
    }
    return cell_bounds_operator_bound_original(law,tree,summary,left,right);
}

void write_bound_certificate_stats(){
    DiagnosticIO io;
    const char*p=std::getenv("S_BOUND_STATS");if(!p)return;
    if(auto f=std::fopen(p,"w")){
        std::fprintf(f,"{\"enabled\":%s,\"shadow\":%s,\"trials\":%llu,\"success\":%llu}\n",
            bound_cert_enabled?"true":"false",bound_cert_shadow?"true":"false",
            (unsigned long long)bound_cert_trials,(unsigned long long)bound_cert_success);std::fclose(f);
    }
}
