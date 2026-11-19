static double diagnostic_io_seconds=0.0;
static unsigned long long diagnostic_io_scopes=0;
static int diagnostic_io_depth=0;
struct DiagnosticIO {
    bool outer;std::chrono::steady_clock::time_point start;
    DiagnosticIO():outer(diagnostic_io_depth++==0){if(outer){start=std::chrono::steady_clock::now();++diagnostic_io_scopes;}}
    ~DiagnosticIO(){--diagnostic_io_depth;if(outer)diagnostic_io_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
};
static void write_research_rank(std::FILE*f,long long value){
    DiagnosticIO io;
    if(std::fwrite(&value,sizeof(value),1,f)!=1)throw std::runtime_error("cannot write research rank trace");
}
