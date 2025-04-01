#pragma once

#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "../../gen_code/hhe.grpc.pb.h"

#include "../../Common.h"

#include "CSP.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using hheproto::AnalystService;
using hheproto::Empty;
using hheproto::CiphertextMsg;
using hheproto::CiphertextResult;

class AnalystServiceCSPClient 
{
    public:
        AnalystServiceCSPClient(shared_ptr<Channel> channel, shared_ptr<BaseCSP> csp)                    
        {
            stub_ = AnalystService::NewStub(channel);
            this->csp = csp;
        }

        /**
        rpc service - Send encrypted result to Analyst
        */
        bool addEncryptedResult(string patientId, string analystId);

    private:
        unique_ptr<AnalystService::Stub> stub_;
        shared_ptr<BaseCSP> csp;
};