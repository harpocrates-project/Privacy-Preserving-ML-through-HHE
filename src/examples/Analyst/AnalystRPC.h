#pragma

#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "../../gen_code/hhe.grpc.pb.h"

#include "Analyst.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerWriter;
using grpc::ServerReaderWriter;
using grpc::Status;

using hheproto::AnalystService;
using hheproto::Empty;
using hheproto::PublicKeyMsg;
using hheproto::CiphertextMsg;
using hheproto::CiphertextResult;

using namespace hheproto;


class AnalystServiceImpl final:public AnalystService::Service
{
    public:
        AnalystServiceImpl(string url, BaseAnalyst* a)
        {
            this->url = url;
            analyst = a;
        }

        /**
        rpc service - get the public key
        */
        Status getPublicKey(ServerContext* context, const Empty* request, PublicKeyMsg* reply) override; 
        
        /**
        rpc service - get the encrypted result from CSP
        */
        Status addEncryptedResult(ServerContext* context, const CiphertextResult* request, Empty* reply) override;    

        void runServer();
        void stopServer();

    private:
        BaseAnalyst* analyst;
        string url;

        void startRPCService();
};
