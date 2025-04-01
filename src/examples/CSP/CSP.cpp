#include "CSP.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// setter
/**
Create a HE key generator
*/
void BaseCSP::setKeyGenerator()
{
    csp_keygen = new KeyGenerator(*context);
}

/**
Create a HE evaluator
*/
void BaseCSP::setEvaluator()
{
    csp_he_eval = new Evaluator(*context);
}

/**
Create a HE Secret key
*/
void BaseCSP::setHESecretKey()
{
    std::cout << "[CSP] Creating a new HE secret key from the context" << std::endl;
    csp_he_sk = csp_keygen->secret_key();
}

/**
Set he_enc_data_processed_map
*/
void BaseCSP::setHHEEncDataProcessedMap(string patientId, string analystId, vector<Ciphertext> ciphertexts)
{
    he_enc_data_processed_map[analystId][patientId] = ciphertexts;
}

// getter
/**
Return the HE key generator
*/
KeyGenerator *BaseCSP::getKeyGenerator()
{
    return csp_keygen;
}

/**
Return the HE evaluator
*/
Evaluator *BaseCSP::getEvaluator()
{
    return csp_he_eval;
}

/**
Return the HE Secret key
*/
SecretKey BaseCSP::getHESecretKey()
{
    return csp_he_sk;
}

/**
Return the Analyst HE Public key
*/
PublicKey BaseCSP::getAnalystHEPublicKey(string analystId)
{
    auto it = analyst_he_pk_map.find(analystId);
    if (it == analyst_he_pk_map.end())
    {
        cerr << "[CSP] Error: Public key not found for AnalystId: " << analystId << endl;
        throw runtime_error("Public key not found");
    }
    return *(it->second);
}

/**
Return the Analyst HE Relin keys
*/
RelinKeys BaseCSP::getAnalystHERelinKeys(string analystId)
{
    auto it = analyst_he_rk_map.find(analystId);
    if (it == analyst_he_rk_map.end())
    {
        cerr << "[CSP] Error: Relin keys not found for AnalystId: " << analystId << endl;
        throw runtime_error("Relin keys not found");
    }
    return *(it->second);
}

/**
Return the Analyst HE Galois keys
*/
GaloisKeys BaseCSP::getAnalystHEGaloisKeys(string analystId)
{
    auto it = analyst_he_gk_map.find(analystId);
    if (it == analyst_he_gk_map.end())
    {
        cerr << "[CSP] Error: Galois keys not found for AnalystId: " << analystId << endl;
        throw runtime_error("Galois keys not found");
    }
    return *(it->second);
}

/**
Return the User encrypted symmetric key
*/
vector<Ciphertext> BaseCSP::getUserEncryptedSymmetricKey(string analystId)
{
    auto it = enc_sym_key_map.find(analystId);
    if (it == enc_sym_key_map.end())
    {
        cerr << "[CSP] Error: Encrypted symmetric key not found for AnalystId: " << analystId << endl;
        throw runtime_error("Encrypted symmetric key not found");
    }
    return it->second;
}

/**
Return the User encrypted data
*/
vector<vector<uint64_t>>& BaseCSP::getUserEncryptedData(string patientId, string analystId) 
{
    auto it = enc_data_map.find(analystId);
    if (it == enc_data_map.end() || it->second.find(patientId) == it->second.end())
    {
        cerr << "[CSP] Error: Encrypted data not found for AnalystId: " << analystId << endl;
        throw runtime_error("Encrypted data not found");
    }
    return it->second[patientId];
}

/**
Return the HE encrypted data
*/
vector<vector<Ciphertext>>& BaseCSP::getHEEncryptedData(string patientId, string analystId)
{
    auto it = he_enc_data_map.find(analystId);
    if (it == he_enc_data_map.end() || it->second.find(patientId) == it->second.end())
    {
        cerr << "[CSP] Error: HE encrypted data not found for AnalystId: " << analystId << " and PatientId: " << patientId << endl;
        throw runtime_error("HE encrypted data not found");
    }
    return it->second[patientId];
}

/**
Return the Sum of the HE_ENC_Product
*/
vector<Ciphertext> BaseCSP::getHESumEncProduct(string patientId, string analystId)
{
    auto it = he_sum_enc_product_map.find(analystId);
    if (it == he_sum_enc_product_map.end() || it->second.find(patientId) == it->second.end())
    {
        cerr << "[CSP] Error: Sum of HE_ENC_Product not found for AnalystId: " << analystId << " and PatientId: " << patientId << endl;
        throw runtime_error("Sum of HE_ENC_Product not found");
    }
    return it->second[patientId];
}

/**
Return the encrypted result calculated by CSP via HHE decomposition and evaluation
*/

int BaseCSP::getEncryptedResultBytes(string patientId, string analystId, seal_byte *&buffer, int index)
{
    //Ciphertext encrypted_sum_vec = he_sum_enc_product_map[analystId][patientId].at(index);
    Ciphertext encrypted_sum_vec = getHESumEncProduct(patientId, analystId).at(index);

    int encrypted_sum_vec_size = encrypted_sum_vec.save_size();
    buffer = new seal_byte[encrypted_sum_vec_size];
    encrypted_sum_vec.save(buffer, encrypted_sum_vec_size);

    cout << "[CSP] Serialised encrypted result for Analyst (AnalystId: " << analystId << ", size: " << encrypted_sum_vec_size << ")" << endl;
    print_seal_bytes(buffer);

    return encrypted_sum_vec_size;
}

/**
Return the HE encrypted processed data
*/
vector<Ciphertext>& BaseCSP::getHEEncDataProcessedMapValue(string patientId, string analystId)
{
    auto it = he_enc_data_processed_map.find(analystId);
    if (it == he_enc_data_processed_map.end() || it->second.find(patientId) == it->second.end())
    {
        cerr << "[CSP] Error: HE encrypted processed data not found for AnalystId: " << analystId << " and PatientId: " << patientId << endl;
        throw runtime_error("HE encrypted processed data not found");
    }
    return it->second[patientId];
}


/**
Return the first value of encrypted weights map
*/
Ciphertext BaseCSP::getEncWeightsMapFirstValue(string analystId)
{
    auto it = enc_weights_map.find(analystId);
    if (it == enc_weights_map.end() || it->second.empty())
    {
        cerr << "[CSP] Error: Encrypted weights not found for AnalystId: " << analystId << endl;
        throw runtime_error("Encrypted weights not found");
    }
    return it->second[0];
}

/**
Return the CSP Relin keys value
*/
RelinKeys BaseCSP::getCSPHERelinKeysMapValue(string analystId)
{
    auto it = csp_he_rk_map.find(analystId);
    if (it == csp_he_rk_map.end())
    {
        cerr << "[CSP] Error: CSP Relin keys not found for AnalystId: " << analystId << endl;
        throw runtime_error("CSP Relin keys not found");
    }
    return *(it->second);
}

/**
Return the CSP Galois keys value
*/
GaloisKeys BaseCSP::getCSPHEGaloisKeysMapValue(string analystId)
{
    auto it = csp_he_gk_map.find(analystId);
    if (it == csp_he_gk_map.end())
    {
        cerr << "[CSP] Error: CSP Galois keys not found for AnalystId: " << analystId << endl;
        throw runtime_error("CSP Galois keys not found");
    }
    return *(it->second);
}

/** 
Return the HE encrypted product data
*/
vector<Ciphertext>& BaseCSP::getHEEncProductMapValue(string patientId, string analystId) {
    auto it = he_enc_product_map.find(analystId);
    if (it == he_enc_product_map.end() || it->second.find(patientId) == it->second.end()) {
        cerr << "[CSP] Error: HE encrypted product data not found for AnalystId: " << analystId << " and PatientId: " << patientId << endl;
        throw runtime_error("HE encrypted product data not found");
    }
    return it->second[patientId];
}

/** 
Return the HE sum encrypted product data
 */
vector<Ciphertext>& BaseCSP::getHESumEncProductMapValue(string patientId, string analystId) {
    auto it = he_sum_enc_product_map.find(analystId);
    if (it == he_sum_enc_product_map.end() || it->second.find(patientId) == it->second.end()) {
        cerr << "[CSP] Error: HE sum encrypted product data not found for AnalystId: " << analystId << " and PatientId: " << patientId << endl;
        throw runtime_error("HE sum encrypted product data not found");
    }
    return it->second[patientId];
}

// functions
/*
Helper function to print the first ten bytes of the seal_byte input.
*/
void BaseCSP::print_seal_bytes(seal_byte *buffer)
{
    for (int i = 0; i < 10; i++)
    {
        cout << (int)buffer[i] << ' ';
    }
    cout << "... ..." << endl;
}

/**
Helper function to print ciphertext
*/
void BaseCSP::print_Ciphertext(Ciphertext input)
{
    seal_byte *buffer = nullptr;

    int input_size = input.save_size();
    buffer = new seal_byte[input_size];
    input.save(buffer, input_size);
    print_seal_bytes(buffer);
    delete[] buffer;
}

/**
Helper function to print the ciphertext vector
*/
void BaseCSP::print_vec_Ciphertext(vector<Ciphertext> input, size_t size)
{
    for (int i = 0; i < size; i++)
    {
        print_Ciphertext(input[i]);
        cout << "input_size: " << input[i].save_size() << endl;
    }
}

/**
Set up HE parameters
*/
void BaseCSP::heInit()
{
    setKeyGenerator();
    setEvaluator();
    setHESecretKey();
}

/**
HHE decomposition
*/
bool BaseCSP::decompose(string patientId, string analystId, int inputLen)
{
    try
    {
        cout << "[CSP] Making a PASTA_SEAL HHE object based on the CSP's HE sk and Analyst's HE pk, rk, gk (Analyst Id: " << analystId << ")" << endl;

        pasta::PASTA_SEAL HHE(context,
                              getAnalystHEPublicKey(analystId),
                              getHESecretKey(),
                              getAnalystHERelinKeys(analystId),
                              getAnalystHEGaloisKeys(analystId));

        cout << "[CSP] Decomposition: CSP does HHE decomposition to turn User's symmetric input into HE input" << endl;

        auto start = high_resolution_clock::now();

        // Perform decomposition
        performDecomposition(patientId, analystId, HHE);

        // Perform masking
        performMasking(patientId, analystId, inputLen, HHE);

        // Perform flattening
        performFlattening(patientId, analystId, HHE);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        cout << "Total decompose time: " << duration.count() << " ms" << endl;

        cout << "The HHE decomposition postprocessing result is " << endl; // vi_he_processed
        for (Ciphertext record : getHEEncDataProcessedMapValue(patientId, analystId))
            print_Ciphertext(record);

        cout << "[CSP] Decomposition completed" << endl;
        return true;
    }
    catch (const runtime_error &e)
    {
        cerr << "[CSP] Error during decomposition: " << e.what() << endl;
        return false;
    }
    catch (...)
    {
        cerr << "[CSP] Unknown error during decomposition" << endl;
        return false;
    }
}

void BaseCSP::performDecomposition(string patientId, string analystId, pasta::PASTA_SEAL &HHE)
{
    for (vector<uint64_t> &record : getUserEncryptedData(patientId, analystId))
    {
        // Perform decomposition and store the result in a local variable
        vector<Ciphertext> local_result = HHE.decomposition(record, getUserEncryptedSymmetricKey(analystId), config::USE_BATCH);

        he_enc_data_map[analystId][patientId].push_back(std::move(local_result));
    }
}

void BaseCSP::performMasking(string patientId, string analystId, int inputLen, pasta::PASTA_SEAL &HHE)
{
    size_t rem = inputLen % HHE.get_plain_size();

    if (rem != 0)
    {
        vector<uint64_t> mask(rem, 1);
        {
            for (vector<Ciphertext> record : getHEEncryptedData(patientId, analystId))
            {
                if (record.size() == 0)
                {
                    cout << "Empty record" << endl;
                    continue;
                }

                HHE.mask(record.back(), mask);
            }
        }
    }
}

void BaseCSP::performFlattening(string patientId, string analystId, pasta::PASTA_SEAL &HHE)
{
    Ciphertext tmp;
    for (vector<Ciphertext> record : getHEEncryptedData(patientId, analystId))
    {
        if (record.size() == 0)
        {
            cout << "Empty record" << endl;
            continue;
        }

        HHE.flatten(record,
                    tmp,
                    getCSPHEGaloisKeysMapValue(analystId)); // vi_he_processed = hhe_decomposition = C_prime

        he_enc_data_processed_map[analystId][patientId].push_back(tmp);
    }
}

/**
HHE evaluation
*/
bool CSP_hhe_pktnn_1fc::evaluateModel(string patientId, string analystId, int inputLen)
{
    cout << "[CSP] Evaluating the HE weights on the decomposed HE data" << endl;

    auto start = high_resolution_clock::now();

    try
    {
        Ciphertext tmp;
        for (Ciphertext record : getHEEncDataProcessedMapValue(patientId, analystId))
        {
            sealhelper::packed_enc_multiply(record,
                                            getEncWeightsMapFirstValue(analystId),
                                            tmp,
                                            *getEvaluator());

            he_enc_product_map[analystId][patientId].push_back(tmp);
        }

        Ciphertext tmp1;
        for (Ciphertext record : getHEEncProductMapValue(patientId, analystId))
        {
            cout << "encrypted_product size before relinearization = " << record.size() << endl;
            getEvaluator()->relinearize_inplace(record, getCSPHERelinKeysMapValue(analystId));
            cout << "encrypted_product size after relinearization = " << record.size() << endl;

            // Do encrypted sum on the resulting product vector
            cout << "[CSP] Executing encrypted sum on the encrypted vector" << endl;

            sealhelper::encrypted_vec_sum(record,
                                          tmp1,
                                          *getEvaluator(),
                                          getAnalystHEGaloisKeys(analystId),
                                          inputLen);

            he_sum_enc_product_map[analystId][patientId].push_back(tmp1);
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        cout << "Total evaluation time: " << duration.count() << " ms" << endl;

        print_vec_Ciphertext(getHESumEncProductMapValue(patientId, analystId), getHESumEncProductMapValue(patientId, analystId).size());

        cout << "[CSP] Evaluation completed" << endl;
        return true;
    }
    catch (const runtime_error &e)
    {
        cerr << "[CSP] Error during evaluation: " << e.what() << endl;
        return false;
    }
}

/**
Add Analyst HE Public key on CSP
*/
bool BaseCSP::addAnalystHEPublicKey(string analystId, seal_byte *bytes, int size)
{
try
    {
    cout << "[CSP] Adding Analyst HE Public key (AnalystId: " << analystId << ") and (size=" << size << ")" << endl;
    print_seal_bytes(bytes);

    PublicKey *analyst_he_pk = new PublicKey();
    analyst_he_pk->load(*context, bytes, size);

    analyst_he_pk_map[analystId] = analyst_he_pk;

    return true;
}
    catch (const runtime_error &e)
    {
        cerr << "[CSP] Error adding Analyst HE Public key: " << e.what() << endl;
        return false;
    }
}

/**
Add Analyst HE Galois keys on CSP
*/
bool BaseCSP::addAnalystHERelinKeys(string analystId, seal_byte *bytes, int size)
{
    cout << "[CSP] Adding Analyst HE Relin keys (AnalystId: " << analystId << ") and (size=" << size << ")" << endl;
    print_seal_bytes(bytes);

    RelinKeys *analyst_he_rk = new RelinKeys();
    analyst_he_rk->load(*context, bytes, size);

    analyst_he_rk_map[analystId] = analyst_he_rk;

    return true;
}

/**
Add CSP HE Relin keys
*/
bool BaseCSP::addHERelinKeys(string analystId, seal_byte *bytes, int size)
{
    cout << "[CSP] Adding CSP Relin keys (AnalystId: " << analystId << ") and (size=" << size << ")" << endl;
    print_seal_bytes(bytes);

    RelinKeys *csp_he_rk = new RelinKeys();
    csp_he_rk->load(*context, bytes, size);

    csp_he_rk_map[analystId] = csp_he_rk;

    return true;
}

/**
Add Analyst HE Galois keys on CSP
*/
bool BaseCSP::addAnalystHEGaloisKeys(string analystId, seal_byte *bytes, int size)
{
    cout << "[CSP] Adding Analyst Galois keys (AnalystId: " << analystId << ") and (size=" << size << ")" << endl;
    print_seal_bytes(bytes);

    GaloisKeys *analyst_he_gk = new GaloisKeys();
    analyst_he_gk->load(*context, bytes, size);

    analyst_he_gk_map[analystId] = analyst_he_gk;

    return true;
}

/**
Add CSP HE Galois keys
*/
bool BaseCSP::addHEGaloisKeys(string analystId, seal_byte *bytes, int size)
{
    cout << "[CSP] Adding CSP Galois keys (AnalystId: " << analystId << ") and (size=" << size << ")" << endl;
    print_seal_bytes(bytes);

    GaloisKeys *csp_he_gk = new GaloisKeys();
    csp_he_gk->load(*context, bytes, size);

    csp_he_gk_map[analystId] = csp_he_gk;

    return true;
}

/**
Add User encrypted symmetric key on CSP
*/
bool BaseCSP::addUserEncryptedSymmetricKey(string analystId, vector<seal_byte *> bytes, vector<int> lengths)
{
    cout << "[CSP] Adding User encrypted symmetric key (AnalystId: " << analystId << ")" << endl;

    vector<Ciphertext> keys;

    for (int i = 0; i < bytes.size(); i++)
    {
        Ciphertext *key = new Ciphertext();
        key->load(*context, bytes[i], lengths[i]);
        keys.push_back(*key);
        delete key;
    }

    enc_sym_key_map[analystId] = keys;

    return true;
}

/**
Add User encrypted data on CSP
*/
bool BaseCSP::addUserEncryptedData(string patientId, string analystId, vector<vector<uint64_t>> values)
{
    cout << "[CSP] Adding User encrypted data (PatientId: " << patientId << " AnalystId: " << analystId << ")" << endl;

    enc_data_map[analystId][patientId] = values;
    return true;
}

/**
Add Analyst UUID
*/
bool BaseCSP::addAnalystUUID(string analystId, string analystUUID)
{
    cout << "[CSP] Adding Analyst's UUID (AnalystId: " << analystId << ")" << endl;

    analyst_uuid_map[analystId] = analystUUID;

    return true;
}

bool BaseCSP::addAnalystUUIDtoIDMap(string analystUUID, string analystId)
{
    cout << "[CSP] Adding Analyst's UUID to ID map (AnalystId: " << analystId << ")" << endl;

    analyst_uuid_id_map[analystUUID] = analystId;

    return true;
}

/**
Return the Analyst's UUID
*/
string BaseCSP::getAnalystUUID(string analystId)
{
    return analyst_uuid_map[analystId];
}

string BaseCSP::getAnalystIdfromUUID(string analystId)
{
    return analyst_uuid_id_map[analystId];
}

/**
Add Analyst NN model encrypted weights on CSP
*/
bool BaseCSP::addAnalystEncryptedWeights(string analystId, vector<seal_byte *> bytes, vector<int> size)
{
    cout << "[CSP] Adding ML model encrypted weights (AnalystId: " << analystId << ")" << endl;

    vector<Ciphertext> weights;

    for (int i = 0; i < bytes.size(); i++)
    {
        Ciphertext *weight = new Ciphertext();
        weight->load(*context, bytes[i], size[i]);
        weights.push_back(*weight);
        delete weight;
    }

    enc_weights_map[analystId] = weights;

    return true;
}

/**
Write HHE Decomposition data from memory to a file
*/
bool BaseCSP::writeHHEDecompositionDataToFile(string fileName, vector<Ciphertext>& input)
{
    // fileName = fileName;
    ofstream out(fileName, ios::binary);
    if (!out.is_open())
    {
        //throw ios_base::failure("Failed to open file for writing");
        return false;
    }

    // Save the size of the vector
    size_t size = input.size();
    cout << "The size of the vector is " << size << endl;

    out.write(reinterpret_cast<const char *>(&size), sizeof(size));

    // Save each ciphertext
    for (const auto &ciphertext : input)
    {
        ciphertext.save(out);
    }

    out.close();

    cout << "The HHE decomposition data has been written to file: " << fileName << endl;

    return true;
}

/**
Read HHE Decomposition data from a file
*/
bool BaseCSP::readHHEDecompositionDataFromFile(string fileName, vector<Ciphertext> &output)
{
    ifstream in(fileName, ios::binary);
    if (!in.is_open())
    {
        return false;
    }

    // Read the size of the vector
    size_t size;
    in.read(reinterpret_cast<char *>(&size), sizeof(size));

    // Resize the vector and load each ciphertext

    output.resize(size);
    for (size_t i = 0; i < size; ++i)
    {
        output[i].load(*context, in);
    }

    in.close();

    cout << "Read HHE decomposition data from a file" << endl;
    for (Ciphertext t : output)
        print_Ciphertext(t);

    return true;
}

/**
Convert HHEDecomp data from bytes to Ciphertext
*/
bool BaseCSP::deserializeCiphertexts(const google::protobuf::RepeatedPtrField<std::string> &serializedDataList,
                                     std::vector<Ciphertext> &ciphertexts,
                                     std::string &errorMessage)
{
    try
    {
        // Check if serialized data list is empty
        if (serializedDataList.empty())
        {
            errorMessage = "Received empty HHE decomposition data.";
            return false;
        }

        // Concatenate all strings in the RepeatedPtrField into one continuous string
        std::ostringstream concatenatedData;
        for (const auto &data : serializedDataList)
        {
            concatenatedData << data;
        }

        std::istringstream inputStream(concatenatedData.str());

        // Read the size of the ciphertext array
        size_t arraySize;
        inputStream.read(reinterpret_cast<char *>(&arraySize), sizeof(arraySize));
        if (!inputStream.good())
        {
            errorMessage = "Failed to read the size of the ciphertext array.";
            return false;
        }

        std::cout << "Array size: " << arraySize << std::endl;

        // Resize the vector to hold the ciphertexts
        ciphertexts.resize(arraySize);

        // Deserialize each ciphertext
        for (size_t i = 0; i < arraySize; ++i)
        {
            try
            {
                ciphertexts[i].load(*context, inputStream);
                if (!inputStream.good())
                {
                    errorMessage = "Failed to load a ciphertext from the stream.";
                    return false;
                }
            }
            catch (const std::exception &e)
            {
                errorMessage = "Error deserializing ciphertext: " + std::string(e.what());
                return false;
            }
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in deserializeCiphertexts: " << e.what() << std::endl;
        errorMessage = e.what();
        return false;
    }
}

void BaseCSP::removeHEDecomposeData(string patientId, string analystId) {
    getUserEncryptedSymmetricKey(analystId).clear();
    getUserEncryptedSymmetricKey(analystId).shrink_to_fit();
    enc_sym_key_map.unsafe_erase(analystId);
    cout << "Number of elements in the enc_sym_key_map for analystId: " << analystId << " is " << enc_sym_key_map[analystId].size() << endl;

    auto& data = getUserEncryptedData(patientId, analystId);
    data.clear();
    data.shrink_to_fit();
    enc_data_map[analystId].unsafe_erase(patientId);
    cout << "Number of elements in the enc_data_map for analystId: " << analystId << " is " << enc_data_map[analystId].size() << endl;

    auto& he_data = getHEEncryptedData(patientId, analystId);
    he_data.clear();
    he_data.shrink_to_fit();
    he_enc_data_map[analystId].unsafe_erase(patientId);
    cout << "Number of elements in the he_enc_data_map for analystId: " << analystId << " is " << he_enc_data_map[analystId].size() << endl;

    auto& he_data_processed = getHEEncDataProcessedMapValue(patientId, analystId);
    he_data_processed.clear();
    he_data_processed.shrink_to_fit();
    he_enc_data_processed_map[analystId].unsafe_erase(patientId);
    cout << "Number of elements in the he_enc_data_processed_map for analystId: " << analystId << " is " << he_enc_data_processed_map[analystId].size() << endl;
}

void BaseCSP::removeHEEvaluateData(string patientId, string analystId) {
    auto& he_data_processed = getHEEncDataProcessedMapValue(patientId, analystId);
    he_data_processed.clear();
    he_data_processed.shrink_to_fit();
    he_enc_data_processed_map[analystId].unsafe_erase(patientId);
    cout << "Number of elements in the he_enc_data_processed_map for analystId: " << analystId << " is " << he_enc_data_processed_map[analystId].size() << endl;
    
    auto& he_product = getHEEncProductMapValue(patientId, analystId);
    he_product.clear();
    he_product.shrink_to_fit();
    he_enc_product_map[analystId].unsafe_erase(patientId);
    cout << "Number of elements in the he_enc_product_map for analystId: " << analystId << " is " << he_enc_product_map[analystId].size() << endl;

    auto& he_sum_product = getHESumEncProductMapValue(patientId, analystId);
    he_sum_product.clear();
    he_sum_product.shrink_to_fit();
    he_sum_enc_product_map[analystId].unsafe_erase(patientId);
    cout << "Number of elements in the he_sum_enc_product_map for analystId: " << analystId << " is " << he_sum_enc_product_map[analystId].size() << endl;
}


void CSPParallel_hhe_pktnn_1fc::performDecomposition(string patientId, std::string analystId, pasta::PASTA_SEAL &HHE)
{
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> thread_pool;

    std::cout << "Using up to " << num_threads << " threads for decomposition" << std::endl;

    auto decompose_task = [&](vector<uint64_t> record, vector<vector<Ciphertext>> &he_enc_data, const vector<Ciphertext> &userEncryptedSymmetricKey, size_t index)
    {
        try
        {
            std::cout << "Thread " << std::this_thread::get_id() << " started for record " << index << std::endl;

            // Perform decomposition and store the result in a local variable
            vector<Ciphertext> local_result = HHE.decomposition(record, userEncryptedSymmetricKey, config::USE_BATCH);

            he_enc_data[index] = std::move(local_result);

            std::cout << "Thread " << std::this_thread::get_id() << " finished for record " << index << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception in thread " << std::this_thread::get_id() << " for record " << index << ": " << e.what() << std::endl;
            throw;
        }
        catch (...)
        {
            std::cerr << "Unknown exception in thread " << std::this_thread::get_id() << " for record " << index << std::endl;
            throw;
        }
    };

    auto &he_enc_data = he_enc_data_map[analystId][patientId];
    cout << getUserEncryptedData(patientId, analystId).size() << " records to process for: " << patientId << endl;
    he_enc_data.resize(getUserEncryptedData(patientId, analystId).size());
    const auto &userEncryptedSymmetricKey = getUserEncryptedSymmetricKey(analystId);
    size_t num_of_active_threads = 0;

    const auto &records = getUserEncryptedData(patientId, analystId);
    for (size_t i = 0; i < records.size(); ++i)
    {
        const auto &record = records[i];
        size_t num_of_record_blocks = HHE.get_num_of_blocks(record);

        std::cout << "Processing record " << i << " with " << num_of_record_blocks << " blocks" << std::endl;

        manageThreadPool(thread_pool, num_of_active_threads, num_threads);

        std::cout << "Creating thread for record " << i << std::endl;
        thread_pool.emplace_back(decompose_task, record, std::ref(he_enc_data), std::ref(userEncryptedSymmetricKey), i);
        num_of_active_threads = thread_pool.size() * num_of_record_blocks;
        std::cout << "Current number of active threads: " << num_of_active_threads << std::endl;
    }

    waitForRemainingThreads(thread_pool);

    std::cout << "All threads finished. Printing results." << std::endl;

    cout << records.size() << " records processed" << endl;
    cout << he_enc_data.size() << " records in he_enc_data" << endl;

    for (auto &record : getHEEncryptedData(patientId, analystId))
    {
        if (record.size() == 0)
        {
            std::cerr << "Empty record" << std::endl;
            continue;
        }
        cout << "Record size: " << record.size() << endl;
        print_vec_Ciphertext(record, record.size());
    }

    cout << "[CSP] Initial Decomposition completed" << endl;
}

bool CSPParallel_hhe_pktnn_1fc::evaluateModel(string patientId, string analystId, int inputLen)
{
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> thread_pool;
    bool success = true;

    std::cout << "[CSP] Evaluating the HE weights on the decomposed HE data" << std::endl;

    auto start = high_resolution_clock::now();

    auto &he_enc_product = he_enc_product_map[analystId][patientId];
    const auto &records = getHEEncDataProcessedMapValue(patientId, analystId);
    size_t num_of_active_threads = 0;

    // First loop: Multiply records with encrypted weights
    auto multiply_task = [&](const Ciphertext &record, size_t index)
    {
        try
        {
            std::cout << "Thread " << std::this_thread::get_id() << " started for record " << index << std::endl;
            Ciphertext tmp;
            sealhelper::packed_enc_multiply(record, getEncWeightsMapFirstValue(analystId), tmp, *getEvaluator());

            he_enc_product[index] = tmp;

            std::cout << "Thread " << std::this_thread::get_id() << " finished for record " << index << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception in thread " << std::this_thread::get_id() << " for record " << index << ": " << e.what() << std::endl;
            success = false;
        }
        catch (...)
        {
            std::cerr << "Unknown exception in thread " << std::this_thread::get_id() << " for record " << index << std::endl;
            success = false;
        }
    };

    he_enc_product.resize(records.size());

    for (size_t i = 0; i < records.size(); ++i)
    {
        const auto &record = records[i];

        std::cout << "Processing record " << i << std::endl;

        manageThreadPool(thread_pool, num_of_active_threads, num_threads);

        std::cout << "Creating thread for record " << i << std::endl;
        thread_pool.emplace_back(multiply_task, std::cref(record), i);
        num_of_active_threads = thread_pool.size();
        std::cout << "Current number of active threads: " << num_of_active_threads << std::endl;
    }

    waitForRemainingThreads(thread_pool);

    std::cout << "All threads finished. Performing relinearization and sum." << std::endl;

    // Second loop: Relinearize and sum the resulting product vector
    num_of_active_threads = 0;
    auto &he_sum_enc_product = he_sum_enc_product_map[analystId][patientId];

    auto relinearize_and_sum_task = [&](Ciphertext &record, size_t index)
    {
        try
        {
            std::cout << "Thread " << std::this_thread::get_id() << " started for relinearization and sum of record " << index << std::endl;
            std::cout << "encrypted_product size before relinearization = " << record.size() << std::endl;
            getEvaluator()->relinearize_inplace(record, getCSPHERelinKeysMapValue(analystId));
            std::cout << "encrypted_product size after relinearization = " << record.size() << std::endl;

            // Do encrypted sum on the resulting product vector
            std::cout << "[CSP] Executing encrypted sum on the encrypted vector" << std::endl;
            Ciphertext tmp1;
            sealhelper::encrypted_vec_sum(record, tmp1, *getEvaluator(), getAnalystHEGaloisKeys(analystId), inputLen);

            he_sum_enc_product[index] = tmp1;

            std::cout << "Thread " << std::this_thread::get_id() << " finished for relinearization and sum of record " << index << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception in thread " << std::this_thread::get_id() << " for relinearization and sum of record " << index << ": " << e.what() << std::endl;
            success = false;
        }
        catch (...)
        {
            std::cerr << "Unknown exception in thread " << std::this_thread::get_id() << " for relinearization and sum of record " << index << std::endl;
            success = false;
        }
    };

    he_sum_enc_product.resize(he_enc_product.size());

    for (size_t i = 0; i < he_enc_product.size(); ++i)
    {
        auto &record = he_enc_product[i];

        std::cout << "Processing relinearization and sum for record " << i << std::endl;

        manageThreadPool(thread_pool, num_of_active_threads, num_threads);

        std::cout << "Creating thread for relinearization and sum of record " << i << std::endl;
        thread_pool.emplace_back(relinearize_and_sum_task, std::ref(record), i);
        num_of_active_threads = thread_pool.size();
        std::cout << "Current number of active threads: " << num_of_active_threads << std::endl;
    }

    waitForRemainingThreads(thread_pool);

    if (!success)
    {
        std::cerr << "[CSP] Error during evaluation" << std::endl;
        return false;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "Total evaluation time: " << duration.count() << " ms" << std::endl;

    print_vec_Ciphertext(getHESumEncProduct(patientId, analystId), getHESumEncProduct(patientId, analystId).size());

    std::cout << "[CSP] Evaluation completed" << std::endl;
    return true;
}

// Helper method to manage the thread pool
void CSPParallel_hhe_pktnn_1fc::manageThreadPool(std::vector<std::thread> &thread_pool, size_t &num_of_active_threads, unsigned int num_threads)
{
    if (num_of_active_threads >= num_threads)
    {
        std::cout << "Max number of threads reached. Waiting for current threads to finish." << std::endl;
        waitForRemainingThreads(thread_pool);
        num_of_active_threads = 0;
    }
}

// Helper method to wait for remaining threads to finish
void CSPParallel_hhe_pktnn_1fc::waitForRemainingThreads(std::vector<std::thread> &thread_pool)
{
    std::cout << "Waiting for remaining threads to finish." << std::endl;
    for (auto &th : thread_pool)
    {
        th.join();
        std::cout << "Thread " << th.get_id() << " joined" << std::endl;
    }
    thread_pool.clear();
}