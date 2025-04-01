#include <filesystem>
#include "CSPRPC.h"
using grpc::StatusCode;

/** 
rpc service - Add HE Public keys
*/
Status CSPServiceImpl::addPublicKeys(ServerContext* context, const PublicKeySetMsg* request, Empty* reply)
{
    string analystId = getAnalystId(context->client_metadata());

    reply = new Empty();

    cout << "[CSP Service] Adding Analyst HE keys" << endl;

    string strBuffer;
    seal_byte* buffer = nullptr;
    int length;
   
    // Analyst HE Public key
    strBuffer = request->pk().data();
    length = request->pk().length();

    buffer = new seal_byte[length];

    memcpy(buffer, strBuffer.data(), strBuffer.length());
    csp->addAnalystHEPublicKey(analystId, buffer, length);
    delete[] buffer;

    // Analyst HE Relins key
    strBuffer = request->rk().data();
    length = request->rk().length();
   
    buffer = new seal_byte[length];

    memcpy(buffer, strBuffer.data(), strBuffer.length());
    csp->addAnalystHERelinKeys(analystId, buffer, length);
    delete[] buffer;

    // Analyst Galois key
    strBuffer = request->gk().data();
    length = request->gk().length();

    buffer = new seal_byte[length];

    memcpy(buffer, strBuffer.data(), strBuffer.length());
    csp->addAnalystHEGaloisKeys(analystId, buffer, length);
    delete[] buffer;

    // CSP HE Relins key
    strBuffer = request->csp_rk().data();
    length = request->csp_rk().length();
   
    buffer = new seal_byte[length];

    memcpy(buffer, strBuffer.data(), strBuffer.length());
    csp->addHERelinKeys(analystId, buffer, length);
    delete[] buffer;

    // CSP HE Galois key
    strBuffer = request->csp_gk().data();
    length = request->csp_gk().length();
    
    buffer = new seal_byte[length];

    memcpy(buffer, strBuffer.data(), strBuffer.length());
    csp->addHEGaloisKeys(analystId, buffer, length);
    delete[] buffer;

    // Analyst's UUID
    string analystUUID = request->analystuuid();
    cout << "analyst's UUID: " << analystUUID << endl;
    csp->addAnalystUUID(analystId, analystUUID);
    csp->addAnalystUUIDtoIDMap(analystUUID, analystId);

    return Status::OK;
}

/**
rpc service - Add NN encrypted params (weights)
*/
Status CSPServiceImpl::addMLModel(ServerContext* context, const MLModelMsg* request, Empty* reply)
{
    string analystId = getAnalystId(context->client_metadata());

    reply = new Empty();

    cout << "[CSP Service] Adding Analyst ML model" << endl;

    string strBuffer;
    seal_byte* buffer = nullptr;
    
    vector<seal_byte*> wBytes;
    vector<int> wLengths;

    int len;
    for (int i=0, length; i<request->weights_size(); i++)
    {
        strBuffer = request->weights(i).data();
        length = request->weights(i).length();
	    buffer = new seal_byte[length];
        memcpy(buffer, strBuffer.data(), strBuffer.length());
        wBytes.push_back(buffer);
        wLengths.push_back(length);
        len = length;
    }

    cout<<"[CSP Service] ML model encrypted weights and size (" << len << ")" << endl;
    for (int i = 0; i < 10; i++) 
    {
        cout << (int)buffer[i] << ' ';
    }
    cout << "... ..." << endl;

    csp->addAnalystEncryptedWeights(analystId, wBytes, wLengths);

    delete reply;
    for (auto buf : wBytes) {
        delete[] buf;
    }
    
    return Status::OK;
} 

/**
rpc service - Add User encrypted symmetric key
*/
Status CSPServiceImpl::addEncryptedKeys(ServerContext* context, const EncSymmetricKeysMsg* request, Empty* reply)
{
    string analystId = getAnalystId(context->client_metadata());
 
    reply = new Empty();

    cout << "[CSP Service] Adding User encrypted symmetric key" << endl;
    
    string strBuffer;
    seal_byte* buffer = nullptr;

    vector<seal_byte*> keysBytes;
    vector<int> keysLengths;

    int len;
    for (int i=0, length; i<request->key_size(); i++)
    {
        strBuffer = request->key(i).data();
        length = request->key(i).length();
	    buffer = new seal_byte[length];
        memcpy(buffer, strBuffer.data(), strBuffer.length());
        keysBytes.push_back(buffer);
        keysLengths.push_back(length);

        len = length;
    }
    
    cout<<"[CSP Service] User's encrypted symmetric key and size (" << len << ")" << endl;;
    for (int i = 0; i < 10; i++) 
    {
        cout << (int)buffer[i] << ' ';
    }
    cout << "... ..." << endl;

    //cout << "bytes size: " << keysBytes.size() << endl;
    //cout << "lengths size: " << keysLengths.size() << endl;

    csp->addUserEncryptedSymmetricKey(analystId, keysBytes, keysLengths);

    for (auto buf : keysBytes) {
        delete[] buf;
    }
    
    return Status::OK;
}

/**
rpc service - Add User encrypted data for NN calculation
*/
Status CSPServiceImpl::addEncryptedData(ServerContext* context, const EncSymmetricDataMsg* request, Empty* reply)
{
    vector<vector<uint64_t>> values;

    string analystId = getAnalystId(context->client_metadata());
    cout << "analystId: " << analystId << endl;

    reply = new Empty();

    cout << "[CSP Service] Adding User encrypted data" << endl;
    for (EncSymmetricDataRecord record : request->record())
    {
        vector<uint64_t> record_values(record.value().begin(), record.value().end());
        values.push_back(std::move(record_values));
        //utils::print_vec(record_values, record_values.size(), "record");
    }

    // Receive the patientId from the User
    string patientId = request->patientid();

    int inputLen = 300;

    try
    {
        csp->addUserEncryptedData(patientId, analystId, values);

        // HHE decomposition
        if (!csp->decompose(patientId, analystId, inputLen))
        {
            return Status(StatusCode::INTERNAL, "Failed to decompose data");
        }

        // Write HHE decomposition data from memory to a file
        string analystUUID = csp->getAnalystUUID(analystId);
        string fileName = "./" + patientId + "_" + analystUUID + ".bin"; // This needs to be changed in line with Analyst's ID and User's ID
        if(csp->writeHHEDecompositionDataToFile(fileName, csp->getHEEncDataProcessedMapValue(patientId, analystId))) {
            cout << "Removing decompose data from memory for " << analystId << " " << patientId << endl;
            // remove data from memory
            csp->removeHEDecomposeData(patientId, analystId);
        } else {
            return Status(StatusCode::INTERNAL, "Failed to write HHE decomposition data to file");
        }
    }
    // catching unexpected exceptions in addUserEncryptedData or writeHHEDecompositionDataToFile
    catch (const runtime_error &e)
    {
        cerr << "[CSP] Unexpected error during decomposition: " << e.what() << endl;
        return Status(StatusCode::INTERNAL, e.what());
    }

    return Status::OK;
}

/**
rpc service - Receive data and evaluate date under NN model
*/
Status CSPServiceImpl::evaluateModel(ServerContext* context, const CiphertextBytes* request, Empty* reply)
{
    // Analyst's Id from UUID 
    string analystUUID = request->analystid();
    cout << "analyst's UUID: " << analystUUID << endl;

    string analystId = csp->getAnalystIdfromUUID(analystUUID);
    cout << "analyst's ID: " << analystId << endl;

    string patientId = request->patientid();
    cout << "patient's ID: " << patientId << endl;

    reply = new Empty();

    // Retrieve and deserialize the HHEDecomp data
    vector<Ciphertext> ciphertexts; // HHEDecomp data
    string errorMessage;
    if (!csp->deserializeCiphertexts(request->hhedecomp(), ciphertexts, errorMessage)) {
            cerr << "Deserialization failed: " << errorMessage << endl;
            return Status(StatusCode::DATA_LOSS, errorMessage);
        }

    cout << "Successfully deserialized " << ciphertexts.size() << " ciphertexts." << endl;
    csp->setHHEEncDataProcessedMap(patientId, analystId, ciphertexts);


    // transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    // if (lowerStr != "spo2" && lowerStr != "ecg")
    // {
    //     throw std::runtime_error("Dataset must be either SpO2 or ECG");
    // }
    // int inputLen = 0;
    // if (lowerStr == "spo2")
    // {
    //     inputLen = 300;
    // }
    // if (lowerStr == "ecg")
    // {
    //     inputLen = 128;
    // }
        // }
    int inputLen = 300;

    // HHE evaluation
    if (!csp->evaluateModel(patientId, analystId, inputLen))
    {
        return Status(StatusCode::INTERNAL, "Failed to evaluate model");
    }

    // creates an object that is used to callback the Analyst
    AnalystServiceCSPClient* analystRPCClient = new AnalystServiceCSPClient(grpc::CreateChannel(analystId, grpc::InsecureChannelCredentials()), csp);

    if (analystRPCClient->addEncryptedResult(patientId, analystId))
    {
        cout << "Removing evaluation data from memory for " << analystId << " " << patientId << endl;
        csp->removeHEEvaluateData(patientId, analystId);
    }
    else
    {
        return Status(StatusCode::INTERNAL, "Failed to send the encrypted result to Analyst");
    }

    delete analystRPCClient;

    return Status::OK;
}


Status CSPServiceImpl::evaluateModelFromFile(ServerContext* context, const DataFile* request, Empty* reply)
{
    reply = new Empty();

    string fileName = request->filename();

    // Use std::filesystem to extract the filename without the path
    std::filesystem::path filePath(fileName);
    string fileNameOnly = filePath.filename().string();

    // Read HHE decomposition data from a file
    vector<Ciphertext> ciphertexts;
    if (!csp->readHHEDecompositionDataFromFile(fileName, ciphertexts))
        return Status(StatusCode::DATA_LOSS, "Failed to read HHE decomposition data from file");

    // Find the position of the first underscore
    size_t underscorePos = fileNameOnly.find('_');
    // Find the position of the dot before the file extension
    size_t dotPos = fileNameOnly.find(".bin");

    // Extract the substring before the underscore as patientId
    string patientId = fileNameOnly.substr(0, underscorePos);
    cout << "patient's ID: " << patientId << endl;

    // Extract the substring between the underscore and the dot as analystUUID
    string analystUUID = fileNameOnly.substr(underscorePos + 1, dotPos - underscorePos - 1);
    cout << "analyst's UUID: " << analystUUID << endl;

    string analystId = csp->getAnalystIdfromUUID(analystUUID);
    cout << "analyst's ID: " << analystId << endl;

    csp->setHHEEncDataProcessedMap(patientId, analystId, ciphertexts);

    int inputLen = 300;

    if (!csp->evaluateModel(patientId, analystId, inputLen))
    {
        return Status(StatusCode::INTERNAL, "Failed to evaluate model");
    }

    // creates an object that is used to callback the Analyst
    AnalystServiceCSPClient* analystRPCClient = new AnalystServiceCSPClient(grpc::CreateChannel(analystId, grpc::InsecureChannelCredentials()), csp);

    if (analystRPCClient->addEncryptedResult(patientId, analystId))
    {
        cout << "Removing evaluation data from memory for " << analystId << " " << patientId << endl;
        csp->removeHEEvaluateData(patientId, analystId);
    }
    else
    {
        return Status(StatusCode::INTERNAL, "Failed to send the encrypted result to Analyst");
    }

    delete analystRPCClient;

    return Status::OK; 
}


/**
Get Analyst IP Addr
*/
string CSPServiceImpl::getAnalystId(multimap<grpc::string_ref, grpc::string_ref> metadata)
{
    multimap<grpc::string_ref, grpc::string_ref>::const_iterator metadataIterator = metadata.find("analystid");

    if (metadataIterator != metadata.end())
    {
        string analystId(metadataIterator->second.data(), metadataIterator->second.length());
        return analystId;
    }

    return "notFound";
}

void CSPServiceImpl::runServer()
{   
    listener = new thread(&CSPServiceImpl::startRPCService, this);
}

void CSPServiceImpl::startRPCService()
{
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(-1);

    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(url, grpc::InsecureServerCredentials());
    
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(this);

    // Finally assemble the server.
    unique_ptr<Server> server(builder.BuildAndStart());
    cout << endl << "[CSP Service] RPC Server listening on " << url << endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

int main(int argc,char** argv)
{
    BaseCSP* csp;
    CSPServiceImpl* cspRPC;	

    string url;
    if (argc == 2)
    {
        url = argv[1];
    } 
    else if (argc != 1) 
    {
        cout << "[CSP Service] Wrong number of arguments provided – using default values" << endl;
        url = "localhost:50052";
    } 
    else 
    {
        cout << "[CSP Service] No arguments provided – using default values" << endl;
        url = "localhost:50052";
    }

    //csp = new CSP_hhe_pktnn_1fc();
    csp = new CSPParallel_hhe_pktnn_1fc();
    cspRPC = new CSPServiceImpl(url, csp);

    // Set up HE params
    csp->heInit();   
    
    // Start the csp rpc service
    cspRPC->runServer();

    cout << "[CSP Service] Press Enter to exit";
    cin.get();

    return 0;
}