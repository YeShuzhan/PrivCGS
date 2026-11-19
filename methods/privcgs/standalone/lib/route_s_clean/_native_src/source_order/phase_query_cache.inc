thread_local bool phase_cache_enabled=false,phase_cache_shadow=false;
thread_local std::vector<std::uint16_t> phase_tree_id;
thread_local std::vector<double> phase_marked_table,phase_flux_table;
thread_local std::size_t phase_count=0;
thread_local std::uint64_t phase_marked_queries=0,phase_flux_queries=0;

void initialize_phase_query_cache(const LawView&law,const TreeView&tree){
    phase_cache_enabled=false;phase_cache_shadow=std::getenv("S_PHASE_SHADOW")!=nullptr;
    phase_marked_queries=phase_flux_queries=0;phase_count=0;
    phase_tree_id.clear();phase_marked_table.clear();phase_flux_table.clear();
    std::unordered_map<std::uint64_t,std::uint16_t> ids;ids.reserve(128);
    std::vector<std::uint64_t> masks;masks.reserve(128);
    phase_tree_id.resize(tree.count);
    for(int node=0;node<tree.count;++node){
        const auto mask=tree.parent_mask[node];auto found=ids.find(mask);
        if(found==ids.end()){
            if(masks.size()>=128){phase_tree_id.clear();return;}
            const auto id=static_cast<std::uint16_t>(masks.size());masks.push_back(mask);
            found=ids.emplace(mask,id).first;
        }
        phase_tree_id[node]=found->second;
    }
    phase_count=masks.size();phase_marked_table.resize(phase_count);phase_flux_table.resize(phase_count*phase_count);
    for(std::size_t i=0;i<phase_count;++i){
        phase_marked_table[i]=marked_upper(law,masks[i]);
        for(std::size_t j=0;j<phase_count;++j)
            phase_flux_table[i*phase_count+j]=parent_upper(law,masks[i],masks[j]);
    }
    phase_cache_enabled=true;
}
double cached_marked_upper(const LawView&law,const TreeView&tree,int node){
    if(!phase_cache_enabled)return marked_upper(law,tree.parent_mask[node]);
    ++phase_marked_queries;const double result=phase_marked_table[phase_tree_id[node]];
    if(phase_cache_shadow){const double original=marked_upper(law,tree.parent_mask[node]);
        if(std::memcmp(&result,&original,sizeof(double)))throw std::runtime_error("cached marked maximum differs by bits");}
    return result;
}
double cached_parent_upper(const LawView&law,const TreeView&tree,int left,int right){
    if(!phase_cache_enabled)return parent_upper(law,tree.parent_mask[left],tree.parent_mask[right]);
    ++phase_flux_queries;const double result=phase_flux_table[phase_tree_id[left]*phase_count+phase_tree_id[right]];
    if(phase_cache_shadow){const double original=parent_upper(law,tree.parent_mask[left],tree.parent_mask[right]);
        if(std::memcmp(&result,&original,sizeof(double)))throw std::runtime_error("cached parent maximum differs by bits");}
    return result;
}
void write_phase_query_cache_stats(){
    DiagnosticIO io;
    const char*p=std::getenv("S_PHASE_STATS");if(!p)return;
    if(auto f=std::fopen(p,"w")){
        std::fprintf(f,"{\"enabled\":%s,\"shadow\":%s,\"classes\":%llu,\"marked_queries\":%llu,\"flux_queries\":%llu,\"allocated_bytes\":%llu}\n",
            phase_cache_enabled?"true":"false",phase_cache_shadow?"true":"false",(unsigned long long)phase_count,
            (unsigned long long)phase_marked_queries,(unsigned long long)phase_flux_queries,
            (unsigned long long)(phase_tree_id.capacity()*sizeof(std::uint16_t)+(phase_marked_table.capacity()+phase_flux_table.capacity())*sizeof(double)));
        std::fclose(f);
    }
}
