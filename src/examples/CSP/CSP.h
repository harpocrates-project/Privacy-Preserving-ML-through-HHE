#pragma once
#include <oneapi/tbb/concurrent_unordered_map.h>

#include "../../Common.h"
#include <google/protobuf/repeated_field.h> // Include for RepeatedPtrField
#include <sstream>                         // For string stream operations
#include <iostream> 

class BaseCSP
{
    public:
        BaseCSP() 
        {
            context = get_seal_context(config::plain_mod, config::mod_degree, config::seclevel);
            he_benc = new BatchEncoder(*context);
        }

        // setter
        /** 
        Create a HE key generator 
        */
        void setKeyGenerator(); // csp_keygen
        
        /**
        Create a HE evaluator
        */
        void setEvaluator(); // evaluator

        /**
        Create a HE Secret key
        */
        void setHESecretKey(); // csp_he_sk
              
        /**
        Set he_enc_data_processed_map
        @param[in] analystId   The Analyst IP Addr
        @param[in] ciphertexts The HHE Decomposition data
        */
        void setHHEEncDataProcessedMap(string patientId, string analystId, vector<Ciphertext> ciphertexts);
    

        // getter
        /**
        Return the SEAL context
        */
        shared_ptr<SEALContext> getContext() const {
            return context;
        }

        /**
        Return the HE key generator
        */
        KeyGenerator* getKeyGenerator(); // csp_keygen

        /**
        Return the HE evaluator
        */
        Evaluator* getEvaluator(); // csp_he_eval

        /**
        Return the HE Secret key
        */
        SecretKey& getHESecretKey(); // csp_he_sk

        /**
        Return the Analyst HE Public key
        @param[in] analystId The Analyst IP Addr
        */
        PublicKey& getAnalystHEPublicKey(string analystId); // Analyst_he_pk
        
        /**
        Return the Analyst HE Relin keys
        @param[in] analystId The Analyst IP Addr
        */
        RelinKeys& getAnalystHERelinKeys(string analystId); // Analyst_he_rk
        
        /**
        Return the Analyst HE Galois keys
        @param[in] analystId The Analyst IP Addr
        */
        GaloisKeys& getAnalystHEGaloisKeys(string analystId); // Analyst_he_gk 
        
        /**
        Return the User encrypted symmetric key
        @param[in] analystId The Analyst IP Addr
        */
        vector<Ciphertext>& getUserEncryptedSymmetricKey(string analystId); // Client_hhe_key
       
        /**
        Return the User encrypted data
        @param[in] analystId The Analyst IP Addr
        */       
        vector<vector<uint64_t>>& getUserEncryptedData(string patientId, string analystId);  // vi_se
        
        /**
        Return the HE encrypted data
        @param[in] analystId The Analyst IP Addr
        */
        vector<vector<Ciphertext>>& getHEEncryptedData(string patientId, string analystId); // vi_he

        /**
        Return the encrypted result calculated by CSP via HHE decomposition and evaluation
        */
       int getEncryptedResultBytes(string patientId, string analystId, std::vector<seal_byte>& buffer, int index);       

        /** 
        Return the HE encrypted processed data 
        @param[in] analystId The Analyst IP Addr
        */
        vector<Ciphertext>& getHEEncDataProcessedMapValue(string patientId, string analystId);

        /** 
        Return the first value of encrypted weights map
        @param[in] analystId The Analyst IP Addr
        */
        Ciphertext& getEncWeightsMapFirstValue(string analystId);

        /** 
        Return the CSP Relin keys value
        @param[in] analystId The Analyst IP Addr
        */
        RelinKeys& getCSPHERelinKeysMapValue(string analystId);

        /** 
        Return the CSP Galois keys value
        @param[in] analystId The Analyst IP Addr
        */
        GaloisKeys& getCSPHEGaloisKeysMapValue(string analystId);

        /** 
        Return the Sum of the HE_ENC_Product
        @param[in] analystId The Analyst IP Addr
        */
        vector<Ciphertext>& getHESumEncProduct(string patientId, string analystId);

        /** 
         * Return the HE encrypted product data
         * @param[in] analystId The Analyst IP Addr
         * @param[in] patientId The Patient ID
         */
        vector<Ciphertext>& getHEEncProductMapValue(string patientId, string analystId);

        /**
         * Return the HE sum encrypted product data
         * @param[in] analystId The Analyst IP Addr
         * @param[in] patientId The Patient ID
         */
        vector<Ciphertext>& getHESumEncProductMapValue(string patientId, string analystId);

        /**
        Return the Analyst's UUID
        @param[in] analystId The Analyst IP Addr
        */
        string getAnalystUUID(string analystId); 

        string getAnalystIdfromUUID(string analystId);

        // functions
        /**
        Set up HE parameters
        */
        void heInit();
        
        /**
        Add Analyst HE Public key on CSP 
        @param[in] analystId The Analyst IP Addr
        @param[in] bytes The key bytes
        @param[in] size The key length
        */
        bool addAnalystHEPublicKey(string analystId, seal_byte* bytes, int size);
        
        /**
        Add Analyst HE Relin keys on CSP 
        @param[in] analystId The Analyst IP Addr
        @param[in] bytes The key bytes
        @param[in] size The key length
        */
        bool addAnalystHERelinKeys(string analystId, seal_byte* bytes, int size); 
        
        /**
        Add Analyst HE Galois keys on CSP 
        @param[in] analystId The Analyst IP Addr
        @param[in] bytes The key bytes
        @param[in] size The key length
        */
        bool addAnalystHEGaloisKeys(string analystId, seal_byte* bytes, int size);

        /**
        Add CSP HE Relin keys
        @param[in] analystId The Analyst IP Addr
        @param[in] bytes The key bytes
        @param[in] size The key length
        */
        bool addHERelinKeys(string analystId, seal_byte* bytes, int size); 
        
        /**
        Add CSP HE Galois keys
        @param[in] analystId The Analyst IP Addr
        @param[in] bytes The key bytes
        @param[in] size The key length
        */
        bool addHEGaloisKeys(string analystId, seal_byte* bytes, int size);

        /**
        Add Analyst UUID
        @param[in] analystId The Analyst IP Addr
        */
        bool addAnalystUUID(string analystId, string analystUUID);

        
        bool addAnalystUUIDtoIDMap(string analystUUID, string analystId);

        /**
        Add User encrypted symmetric key on CSP
        @param[in] analystId The Analyst IP Addr
        @param[in] bytes The key bytes
        @param[in] lengths The key length
        */
        bool addUserEncryptedSymmetricKey(string analystId, vector<vector<seal_byte>> bytes, vector<int> lengths);
        
        /**
        Add User encrypted data on CSP
        @param[in] analystId The Analyst IP Addr
        @param[in] values The data values
        */
        bool addUserEncryptedData(string patientId, string analystId, vector <vector<uint64_t>> values);
       
        /**
        Add Analyst NN model encrypted weights on CSP
        @param[in] analystId The Analyst IP Addr
        @param[in] bytes The encrypted weights bytes
        @param[in] size The encrypted weights size
        */
        bool addAnalystEncryptedWeights(string analystId, vector<vector<seal_byte>> bytes, vector<int> size);

        /**
        HHE decomposition
        @param[in] analystId The Analyst IP Addr
        @param[in] inputLen The length of dataset
        */
        bool decompose(string patientId, string analystId, int inputLen);

        // pure virtual function
        /**
        HHE evaluation
        @param[in] analystId The Analyst IP Addr
        @param[in] inputLen The length of dataset
        */
        virtual bool evaluateModel(string patientId, string analystId, int inputLen) = 0;

        /**
        Helper function to print the first ten bytes of the seal_byte input
        */
        void print_seal_bytes(seal_byte* buffer);
        
        /**
        Helper function to print ciphertext
        */
        void print_Ciphertext(Ciphertext input);

        /**
        Helper function to print the ciphertext vector 
        */
        void print_vec_Ciphertext(vector<Ciphertext>& input, size_t size);

        /** 
        Write HHE Decomposition data from memory to a file
        @param[in] fileName  The file name for HHE Decomposition data
        @param[in] input     The HHE Decomposition data
        */
        bool writeHHEDecompositionDataToFile(string fileName, vector<Ciphertext>& input);
        
        /** 
        Read HHE Decomposition data from a file
        @param[in] fileName  The file name for HHE Decomposition data
        */
        bool readHHEDecompositionDataFromFile(string fileName, vector<Ciphertext>& output);

        /** 
        Convert HHEDecomp data from bytes to Ciphertext
        */
        bool deserializeCiphertexts(const google::protobuf::RepeatedPtrField<std::string>& serializedDataList, 
                            std::vector<Ciphertext>& ciphertexts, 
                            std::string& errorMessage);

        void removeHEDecomposeData(string patientId, string analystId);
        void removeHEEvaluateData(string patientId, string analystId);

    private: 

        void performMasking(string patiendId, string analystId, int inputLen, pasta::PASTA_SEAL& HHE);
        
        void performFlattening(string patientId, string analystId, pasta::PASTA_SEAL& HHE);

        shared_ptr<SEALContext> context;
        Evaluator* csp_he_eval;
        BatchEncoder* he_benc;

        KeyGenerator* csp_keygen;
        SecretKey csp_he_sk;

        oneapi::tbb::concurrent_unordered_map<string, unique_ptr<PublicKey>> analyst_he_pk_map;
        oneapi::tbb::concurrent_unordered_map<string, unique_ptr<RelinKeys>> analyst_he_rk_map;
        oneapi::tbb::concurrent_unordered_map<string, unique_ptr<GaloisKeys>> analyst_he_gk_map;
        oneapi::tbb::concurrent_unordered_map<string, unique_ptr<RelinKeys>> csp_he_rk_map;
        oneapi::tbb::concurrent_unordered_map<string, unique_ptr<GaloisKeys>> csp_he_gk_map;

        oneapi::tbb::concurrent_unordered_map<string, string> analyst_uuid_map;
        oneapi::tbb::concurrent_unordered_map<string, string> analyst_uuid_id_map;

        oneapi::tbb::concurrent_unordered_map<string, vector<Ciphertext>> enc_weights_map;  // Analyst's encrypted weights
        oneapi::tbb::concurrent_unordered_map<string, vector<Ciphertext>> enc_sym_key_map; // User's encrypted symmetric keys

        std::mutex data_cleanup_mutex;

    protected:

        virtual void performDecomposition(string patientId, string analystId, pasta::PASTA_SEAL& HHE);

        // Use a nested map to store the data by mapping the analystId and the patientId to the values vector
        oneapi::tbb::concurrent_unordered_map<string, oneapi::tbb::concurrent_unordered_map<string, vector<vector<uint64_t>>>> enc_data_map; // User's encrypted data

        // Use a nested map to store the HE encrypted data by mapping the analystId and the patientId
        oneapi::tbb::concurrent_unordered_map<string, oneapi::tbb::concurrent_unordered_map<string, vector<vector<Ciphertext>>>> he_enc_data_map; // HE encrypted data

        // HHE decomposition postprocessing on the HE encrypted input. (vi_he_processed)
        oneapi::tbb::concurrent_unordered_map<string, oneapi::tbb::concurrent_unordered_map<string, vector<Ciphertext>>> he_enc_data_processed_map;

        // The multiply of vi_he_processed. (encrypted_product)
        oneapi::tbb::concurrent_unordered_map<string, oneapi::tbb::concurrent_unordered_map<string, vector<Ciphertext>>> he_enc_product_map; 

        // The results will be sent to Analyst. (sum of encrypted_product)
        oneapi::tbb::concurrent_unordered_map<string, oneapi::tbb::concurrent_unordered_map<string, vector<Ciphertext>>> he_sum_enc_product_map;
};      

class CSP_hhe_pktnn_1fc : public BaseCSP 
{
    public:
        /**
        The implementation of the pure virtual function 
        @param[in] analystId The Analyst IP Addr
        @param[in] inputLen The length of dataset
        */
        bool evaluateModel(string patientId, string analystId, int inputLen);
};


class CSPParallel_hhe_pktnn_1fc : public CSP_hhe_pktnn_1fc
{
    public:
    bool evaluateModel(string patientId, string analystId, int inputLen) override;

    protected:
        void performDecomposition(std::string patientId, std::string analystId, pasta::PASTA_SEAL& HHE);

    private:
        void manageThreadPool(std::vector<std::thread>& thread_pool, size_t& num_of_active_threads, unsigned int num_threads);
        void waitForRemainingThreads(std::vector<std::thread>& thread_pool);
};