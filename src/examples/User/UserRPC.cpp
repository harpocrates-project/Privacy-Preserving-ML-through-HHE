#include "AnalystServiceUserClient.h"
#include "CSPServiceUserClient.h"
#include "User.h"


int main(int argc,char** argv)
{
    string analystUrl;
    string cspUrl;
    string dataSet;
    int records = 30;

    if (argc == 5) 
    {
        analystUrl = argv[1];
        cspUrl = argv[2];
        dataSet = argv[3];
        records = atoi(argv[4]);
    } else if (argc == 4) 
    {
        analystUrl = argv[1];
        cspUrl = argv[2];
        dataSet = argv[3];
    } 
    else if (argc == 3) 
    {
        analystUrl = argv[1];
        cspUrl = argv[2];
        dataSet = "data/Harpocrates_recordingwise_SIESTA_4percent/c000101_data.txt";
        cout << "Using Default Dataset" << endl;
        //dataSet = "data/arc/SpO2/SpO2_input_cleaned4%.csv";
    } 
    else 
    {
        cout << "[UserRPC] No arguments provided – using default values" << endl;
        analystUrl = "localhost:50051";
        cspUrl = "localhost:50052";
        dataSet = "data/Harpocrates_recordingwise_SIESTA_4percent/c000101_data.txt";
        //dataSet = "data/arc/SpO2/SpO2_input_cleaned4%.csv";
    }
    
    cout << "Analyst: " << analystUrl << endl;
    cout << "CSP URL: " << cspUrl << endl;
    cout << "Data set: " << dataSet << endl;
    cout << "Will encrypt " << records << " records from the Dataset" << endl;

    User* user = new User();

    // Create a gRPC channel for our stub
    AnalystServiceUserClient AnalystRPCClient(
      grpc::CreateChannel(analystUrl, grpc::InsecureChannelCredentials()));

    CSPServiceUserClient CSPRPCClient(
      grpc::CreateChannel(cspUrl, grpc::InsecureChannelCredentials()), user);  


    // Find Patient ID
    cout<<"=============================="<<endl;
    size_t found = dataSet.find_last_of("/");
    string fileName = dataSet.substr(found+1);
    cout << "file: " << fileName << endl;
    // Find the patient id
    size_t found1 = fileName.find("_");
    string patientID = fileName.substr(0,found1);
    cout << "patientID: " << patientID << endl;
   
    cout<<"=============================="<<endl;
    // Set up a data set name for NN calculation
    //string dataset = "SpO2"; // dataset must be either "SpO2" or "ECG"
    user->setDataSet(dataSet);
    // Load the data set and label for NN calculation
    user->loadDataAndLabel(dataSet);
    // Create user's symmetric key which will be used for data encryption;
    user->setSymmetricKey();
    // Encrypt user data via symmetric key algorithm
    user->encryptData(user->getSymmetricKey(), records);

    cout << "=====================" << endl;    
    cout << "[UserRPC] Receiving Analyst HE Public key" << endl;
    seal_byte* buffer = nullptr;
    int length = AnalystRPCClient.getPublicKey(buffer);
    cout << length << endl;

    if (length == 0) 
    {
        cerr << "[UserRPC] Error: Analyst HE Public key not found" << endl;
        return 1;
    }

    cout << "The Analyst HE Public key (AnalystId: " << analystUrl <<")" << endl;
    for (int i = 0; i < 10; i++) 
    {
        cout << (int)buffer[i] << ' ';
    }
    cout << "... ..." << endl;  

    //  Encrypt user symmetric key via HE
    user->encryptSymmetricKey(user->getSymmetricKey(),
                              buffer, 
                              length);

    // (Check) Computing in plain on 1 input vector
    // user->computingCheck();

    // User sends his encrypted symmetric key to the csp
    CSPRPCClient.addEncryptedKeys(analystUrl);
    
    // User sends his encrypted data to the csp
    CSPRPCClient.addEncryptedData(analystUrl, patientID);

    return 0;
}