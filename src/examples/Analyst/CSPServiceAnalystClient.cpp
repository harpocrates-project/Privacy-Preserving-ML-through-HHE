#include "CSPServiceAnalystClient.h"

/**
rpc service - Send HE Public keys to CSP
*/
bool CSPServiceAnalystClient::addPublicKeys(string analystUUID) 
{
    PublicKeySetMsg request;
    Empty reply;
    ClientContext context;

    cout << "[CSPServiceAnalystClient] Sending HE keys to CSP" << endl;

    context.AddMetadata("analystid", analystId); 

    // Analyst HE Public key
    seal_byte* buffer = nullptr;
    int size = analyst->getPublicKeyBytes(buffer);
    request.mutable_pk()->set_data(buffer, size);
    request.mutable_pk()->set_length(size);

    // Analyst HE Relins keys 
    size = analyst->getRelinKeyBytes(buffer);
    request.mutable_rk()->set_data(buffer, size);
    request.mutable_rk()->set_length(size);

    // Analyst HE Galois keys
    size = analyst->getGaloisKeyBytes(buffer);
    request.mutable_gk()->set_data(buffer, size);
    request.mutable_gk()->set_length(size);

    // CSP HE Relins keys
    size = analyst->getCSPRelinKeyBytes(buffer);
    request.mutable_csp_rk()->set_data(buffer, size);
    request.mutable_csp_rk()->set_length(size);

    // CSP HE Galois keys
    size = analyst->getCSPGaloisKeyBytes(buffer);
    request.mutable_csp_gk()->set_data(buffer, size);
    request.mutable_csp_gk()->set_length(size);

    // Analyst's UUID
    request.set_analystuuid(analystUUID);

    // Sends the HE Public keys to CSP
    Status status = stub_->addPublicKeys(&context, request, &reply);

    if (status.ok()) 
    {
      cout << dec << "[CSPServiceAnalystClient] Successfully uploaded HE keys to CSP" << endl;
      return true;
    } 
    else 
    {
      cout << status.error_code() << ": " << status.error_message() << endl;
      return false;
    }
}

/** 
rpc service - Send the NN encrypted params to CSP
*/
bool CSPServiceAnalystClient::addMLModel() 
{
    MLModelMsg request;
    Empty reply;
    ClientContext context;

    cout << dec << "[CSPServiceAnalystClient] Sending ML model to CSP" << endl;

    context.AddMetadata("analystid", analystId);
                           
    int wNumber = analyst->getEncryptedWeights().size();
    seal_byte* wBytes;

    for (int i=0, size; i<wNumber; i++)
    {
        size = analyst->getEncryptedWeightsBytes(wBytes, i);

        //cout << "[CSPServiceAnalystClient] Weights " << i << " size " << size << endl;
        hheproto::CiphertextMsg* weights = request.add_weights();
        weights->set_data(wBytes, size);
        weights->set_length(size);
    }

    // Send the NN encrypted params to CSP
    Status status = stub_->addMLModel(&context, request, &reply);

    if (status.ok()) 
    {
      cout << dec << "[CSPServiceAnalystClient] Successfully uploaded ML model to CSP" << endl;
      return true;
    } 
    else 
    {
      cout << status.error_code() << ": " << status.error_message() << endl;
      return false;
    }
}
