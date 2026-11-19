    struct CachedCertificateCell {
        Cell cell;
        int children[3]={-1,-1,-1};
        int raw_child_index=-1;
        int child_count=0;
        int expanded=0;
    };
    static constexpr std::size_t CERT_CACHE_CAP=131072;
    static constexpr int CERT_CACHE_DEPTH=14;
    mutable std::vector<CachedCertificateCell> certificate_prefix_cache_;
    mutable bool certificate_prefix_shadow_=false;
    mutable std::uint64_t certificate_prefix_calls_=0,certificate_prefix_hits_=0;
    mutable std::uint64_t certificate_prefix_splits_=0,certificate_prefix_fallbacks_=0;

    bool cached_certificate_child(int& cache_index,int left,int right,Cell& parent,int& raw_index) const {
        if(cache_index<0)return false;
        const int old_index=cache_index;
        if(certificate_prefix_cache_[old_index].expanded==0){
            split_specs(tree_,parent,certificate_specs_);
            if(certificate_prefix_cache_.size()+certificate_specs_.size()>CERT_CACHE_CAP){
                certificate_prefix_cache_[old_index].expanded=-1;
            }else{
                const std::int64_t parent_count=cell_count(tree_,parent.left,parent.right,parent.left==parent.right);
                const std::uint32_t split_hash=extend_path(parent.path_hash,0x7FFFFFFFU);
                const double winner_threshold=uniform53(process_seed_,split_hash,0)*static_cast<double>(parent_count);
                std::int64_t cumulative=0;int winner=static_cast<int>(certificate_specs_.size())-1;
                for(int i=0;i<static_cast<int>(certificate_specs_.size());++i){
                    cumulative+=certificate_specs_[i].count;
                    if(winner_threshold<static_cast<double>(cumulative)){winner=i;break;}
                }
                std::uint64_t draw_index=1;
                for(int i=0;i<static_cast<int>(certificate_specs_.size());++i){
                    const ChildSpec&spec=certificate_specs_[i];
                    const double maximum=i==winner?parent.maximum:truncated_gumbel(
                        uniform53(process_seed_,split_hash,draw_index++),std::log(static_cast<double>(spec.count)),parent.maximum);
                    const double upper=cell_upper(law_,tree_,spec.left,spec.right);
                    CachedCertificateCell child;
                    child.cell=Cell{spec.left,spec.right,extend_path(parent.path_hash,static_cast<std::uint32_t>(spec.raw_index)),
                        maximum,upper+temperature_*maximum,0};
                    child.raw_child_index=spec.raw_index;
                    const int next=static_cast<int>(certificate_prefix_cache_.size());
                    certificate_prefix_cache_.push_back(child);
                    certificate_prefix_cache_[old_index].children[i]=next;
                }
                certificate_prefix_cache_[old_index].child_count=static_cast<int>(certificate_specs_.size());
                certificate_prefix_cache_[old_index].expanded=1;++certificate_prefix_splits_;
            }
        }
        if(certificate_prefix_cache_[old_index].expanded!=1)return false;
        int chosen=-1;
        for(int i=0;i<certificate_prefix_cache_[old_index].child_count;++i){
            const int child_index=certificate_prefix_cache_[old_index].children[i];
            const auto&child=certificate_prefix_cache_[child_index];
            const ChildSpec spec{child.cell.left,child.cell.right,child.cell.left==child.cell.right,child.raw_child_index,0};
            if(child_contains_dyad(spec,left,right)){
                if(chosen>=0)throw std::logic_error("cached certificate children overlap");
                chosen=child_index;
            }
        }
        if(chosen<0)throw std::logic_error("cached certificate lost dyad");
        cache_index=chosen;parent=certificate_prefix_cache_[chosen].cell;
        raw_index=certificate_prefix_cache_[chosen].raw_child_index;++certificate_prefix_hits_;
        return true;
    }

    void fill_compact_certificate(int left,int right,CompactSerialPathCertificate&output) const {
        if(certificate_prefix_cache_.empty()){
            certificate_prefix_shadow_=std::getenv("S_CERT_SHADOW")!=nullptr;
            certificate_prefix_cache_.reserve(CERT_CACHE_CAP);
            CachedCertificateCell root;root.cell=root_;certificate_prefix_cache_.push_back(root);
        }
        ++certificate_prefix_calls_;
        fill_compact_certificate_cached(left,right,output);
        if(certificate_prefix_shadow_){
            verify_lazy_certificate(output,left,right);
        }
    }

    void write_certificate_cache_stats() const {
    DiagnosticIO io;
        const char*p=std::getenv("S_CERT_STATS");if(!p)return;
        if(auto f=std::fopen(p,"w")){
            std::fprintf(f,"{\"shadow\":%s,\"calls\":%llu,\"prefix_hits\":%llu,\"cached_splits\":%llu,\"fallback_suffixes\":%llu,\"cells\":%llu,\"capacity_bytes\":%llu,\"depth_limit\":%d,\"cell_limit\":%llu}\n",
                certificate_prefix_shadow_?"true":"false",(unsigned long long)certificate_prefix_calls_,
                (unsigned long long)certificate_prefix_hits_,(unsigned long long)certificate_prefix_splits_,
                (unsigned long long)certificate_prefix_fallbacks_,(unsigned long long)certificate_prefix_cache_.size(),
                (unsigned long long)(certificate_prefix_cache_.capacity()*sizeof(CachedCertificateCell)),CERT_CACHE_DEPTH,
                (unsigned long long)CERT_CACHE_CAP);std::fclose(f);
        }
    }
