#include "AnalystRPC.h"
#include "CSPServiceAnalystClient.h"

/**
rpc service - get the public key
*/
Status AnalystServiceImpl::getPublicKey(ServerContext* context, const Empty* request, PublicKeyMsg* reply)
{
    seal_byte* buffer = nullptr;

    cout << "[Analyst Service] Sending HE Public key to User"<< endl;

    // get the analyst HE Public key size
    int size = analyst->getPublicKeyBytes(buffer);

    reply->set_data(buffer, size);
    reply->set_length(size);

    return Status::OK;
} 

/**
rpc service - get the encrypted result from CSP
*/
Status AnalystServiceImpl::addEncryptedResult(ServerContext* context, const CiphertextResult* request, Empty* reply)
{
    *reply = Empty();

    // Get the patientId from the request and print it
    const std::string& patientId = request->patientid();
    cout << "[Analyst Service] Adding and decrypting the result from CSP for patient: " << patientId << endl;

    //std::vector<std::vector<seal_byte>> resultsBytes;
    //std::vector<int> resultsLengths;

    for (int i = 0; i < request->result_size(); ++i)
    {
        //const std::string& strBuffer = request->result(i).data();
        int length = request->result(i).length();

        std::vector<seal_byte> buffer(length);
        memcpy(buffer.data(), request->result(i).data().c_str(), length);

        //resultsBytes.push_back(std::move(buffer));
        //resultsLengths.push_back(length);

        // Analyst calls decryptData() to decrypt the encrypted result
        analyst->decryptData(patientId, buffer.data(), length);
    }

    // Write predictions to file
    analyst->writePredictionsToFile(patientId);

    return Status::OK;
}


void AnalystServiceImpl::runServer()
{
    std::thread listener(&AnalystServiceImpl::startRPCService, this);
    listener.join();
}

void AnalystServiceImpl::startRPCService()
{
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    // Create an instance of factory ServerBuilder class
    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(-1);

    // Specify the address and port we want to use to listen for client requests using the builder’s AddListeningPort() method.
    builder.AddListeningPort(url,grpc::InsecureServerCredentials());

    // Register our service implementation with the builder.
    builder.RegisterService(this);

    // Call BuildAndStart() on the builder to create and start an RPC server for our service.
    unique_ptr<Server> server(builder.BuildAndStart());
    cout << "[Analyst Service] RCP Server listening on  " << url << endl
         << "[Analyst Service] Press Ctrl+C to exit" << endl;

    // Call Wait() on the server to do a blocking wait until process is killed or Shutdown() is called
    server->Wait();
}

int main(int argc,char** argv)
{
    BaseAnalyst* analyst;
    AnalystServiceImpl* analystRPC; 
    CSPServiceAnalystClient* cspClient;	

    string url;
    string cspUrl;
    string keyStore = ".analyst_keys";
    string idStore = ".analyst_id";

    if (argc == 3)
    {
        url = argv[1];
        cspUrl = argv[2];
    } 
    else if (argc != 1)
    {
        cout << "[Analyst Service] Wrong number of arguments provided – using default values" << endl;
        url = "localhost:50051";
        cspUrl = "localhost:50052";
    } 
    else 
    {
        cout << "[Analyst Service] Wrong number of arguments provided – using default values" << endl;
        url = "localhost:50051";
        cspUrl = "localhost:50052";
    }

    // Create objs of analyst and analystRPC
    analyst = new Analyst_hhe_pktnn_1fc();
    analystRPC = new AnalystServiceImpl(url, analyst);

    ChannelArguments args;
    args.SetMaxSendMessageSize(-1);
    cspClient = new CSPServiceAnalystClient(grpc::CreateCustomChannel(cspUrl, grpc::InsecureChannelCredentials(), args), analyst, url);

    string dataset = "SpO2"; // dataset must be either "SpO2" or "ECG"
    analyst->setDataSet(dataset);

    if (analyst->loadHEKeys(keyStore) < 0) {
        analyst->generateHEKeys(keyStore);  // Set up HE key
    }  
    
    analyst->setEncryptor();    // Set up HE encryptor
    analyst->setDecryptor();    // set up HE decryptor

    // For HHE_PocketNN_1FC calculation
    analyst->NNModelEncryption(analyst->getDataSet());

    // Should send public keys to cloud provider
    cspClient->addPublicKeys(analyst->getOrGenerateID(idStore));

    // Now send encrpyted model data to cloud provider
    cspClient->addMLModel();

    analystRPC->runServer();    // Start Analyst server
    return 0;
}
