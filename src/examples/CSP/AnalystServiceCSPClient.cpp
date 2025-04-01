#include "AnalystServiceCSPClient.h"

/**
rpc service - Send encrypted result to Analyst
*/
bool AnalystServiceCSPClient::addEncryptedResult(string patientId, string analystId) 
{
    CiphertextResult request;
    Empty reply;
    ClientContext context;
    
    cout << "[AnalystServiceCSPClient] Sending the encrypted result to Analyst (AnalystId: " << analystId << ")" << endl;

    // Set the patientId in the request
    request.set_patientid(patientId);

    // get encrypted results
    int resultsNumber = csp->getHESumEncProduct(patientId, analystId).size();
    vector<seal_byte> buffer;

    for (int i=0; i<resultsNumber; i++)
    {
      int size = csp->getEncryptedResultBytes(patientId, analystId, buffer, i);
      hheproto::CiphertextMsg* result = request.add_result();
      result->set_data(buffer.data(), size);
      result->set_length(size);
    }

    // Send the encrypted result to Analyst
    Status status = stub_->addEncryptedResult(&context, request, &reply);

    if (status.ok()) 
    {
      cout << "[AnalystServiceCSPClient] Successfully sent the encrypted result to Analyst (AnalystId: " << analystId << ")" << endl;
      return true;
    } 
    else 
    {
      cout << status.error_code() << ": " << status.error_message() << endl;
      return false;
    }   
}