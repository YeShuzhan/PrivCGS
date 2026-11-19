    mutable std::uint64_t lazy_suffix_nodes_ = 0;
    mutable std::uint64_t lazy_priority_demands_ = 0;
    mutable std::uint64_t lazy_priority_computations_ = 0;
    mutable std::uint64_t lazy_priority_ready_hits_ = 0;
    mutable std::size_t lazy_maximum_depth_ = 0;

    double materialize_compact_priority(
        CompactSerialPathCertificate& certificate, int index
    ) const {
        ++lazy_priority_demands_;
        const std::size_t i = static_cast<std::size_t>(index);
        if (!certificate.priority_ready[i]) {
            const double maximum = certificate.priority[i];
            const double upper = cell_upper(
                law_, tree_, certificate.cell_left[i], certificate.cell_right[i]
            );
            certificate.priority[i] = upper + temperature_ * maximum;
            certificate.priority_ready[i] = 1;
            ++lazy_priority_computations_;
        } else {
            ++lazy_priority_ready_hits_;
        }
        return certificate.priority[i];
    }

    void verify_lazy_certificate(const CompactSerialPathCertificate& value,
                                 int left, int right) const {
        CompactSerialPathCertificate expected;
        fill_compact_certificate_original(left, right, expected);
        if (value.raw_child_index != expected.raw_child_index ||
            value.priority.size() != expected.priority.size())
            throw std::runtime_error("lazy certificate path differs from eager original");
        for (std::size_t i = 0; i < value.priority.size(); ++i) {
            double actual = value.priority[i];
            if (!value.priority_ready[i]) {
                const double upper = cell_upper(
                    law_, tree_, value.cell_left[i], value.cell_right[i]
                );
                actual = upper + temperature_ * value.priority[i];
            }
            if (std::memcmp(&actual, &expected.priority[i], sizeof(double)))
                throw std::runtime_error("lazy certificate priority bits differ from eager original");
        }
    }

    template<class ReadyRange>
    void write_lazy_priority_stats(const ReadyRange& storage) const {
        const char* path = std::getenv("S_LAZY_STATS");
        if (!path) return;
        std::uint64_t extra_payload = 0;
        for (const auto& entry : storage) {
            const auto& certificate = entry->certificate;
            extra_payload += sizeof(int) * (certificate.cell_left.capacity() + certificate.cell_right.capacity());
            extra_payload += sizeof(unsigned char) * certificate.priority_ready.capacity();
        }
        struct OriginalCertificateLayout {
            std::vector<double> priority;
            std::vector<int> raw_child_index;
        };
        const std::uint64_t extra_objects = storage.size() *
            (sizeof(CompactSerialPathCertificate) - sizeof(OriginalCertificateLayout));
        if (auto file = std::fopen(path, "w")) {
            std::fprintf(file,
                "{\"shadow\":%s,\"suffix_nodes\":%llu,\"priority_demands\":%llu,"
                "\"computed_priorities\":%llu,\"ready_hits\":%llu,\"maximum_depth\":%llu,"
                "\"ready_pool_objects\":%llu,\"extra_vector_capacity_bytes\":%llu,"
                "\"extra_object_bytes\":%llu,\"allocator_overhead_included\":false}\n",
                certificate_prefix_shadow_ ? "true" : "false",
                static_cast<unsigned long long>(lazy_suffix_nodes_),
                static_cast<unsigned long long>(lazy_priority_demands_),
                static_cast<unsigned long long>(lazy_priority_computations_),
                static_cast<unsigned long long>(lazy_priority_ready_hits_),
                static_cast<unsigned long long>(lazy_maximum_depth_),
                static_cast<unsigned long long>(storage.size()),
                static_cast<unsigned long long>(extra_payload),
                static_cast<unsigned long long>(extra_objects));
            std::fclose(file);
        }
    }
