    mutable std::uint64_t replay_children_scanned_ = 0;
    mutable std::uint64_t replay_truncated_omitted_ = 0;
    mutable std::uint64_t replay_truncated_computed_ = 0;
    void write_replay_demand_stats() const {
        const char* path = std::getenv("S_DEMAND_STATS");
        if (!path) return;
        if (auto f=std::fopen(path,"w")) {
            std::fprintf(f,"{\"scanned_children\":%llu,\"omitted_truncated\":%llu,\"computed_truncated\":%llu}\n",
                static_cast<unsigned long long>(replay_children_scanned_),
                static_cast<unsigned long long>(replay_truncated_omitted_),
                static_cast<unsigned long long>(replay_truncated_computed_));
            std::fclose(f);
        }
    }
