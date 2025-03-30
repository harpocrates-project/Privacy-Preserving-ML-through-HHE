#pragma once

#include "../../Common.h"
#include <map>
#include <mutex>

class BaseAnalyst
{
    public:
        BaseAnalyst()
        {
		    context = get_seal_context(config::plain_mod, config::mod_degree, config::seclevel);
            setKeyGenerator();
            setBatchEncoder();	
	    }

        // setter
        /** 
        Create a HE key generator 
        */
        void setKeyGenerator(); // analyst_keygen
       
        /**
        Create a batch encoder
        */
        void setBatchEncoder(); // analyst_he_benc
       
        /** 
        Create a HE Secret key
        */
        void setHESecretKey(); // analyst_he_sk
       
        /**
        Create a HE Public key
        */
        void setHEPublicKey(); // analyst_he_pk
       
        /** 
        Create HE Relin keys
        */
        void setHERelinKeys(); // analyst_he_rk
       
        /** 
        Create HE Galois keys
        */
        void setHEGaloisKeys();  // analyst_he_gk
       
        /**
        Create a HE encryptor
        */
        void setEncryptor(); // encryptor
       
        /** 
        Create a HE evaluator
        */
        void setEvaluator(); // evaluator
       
        /** 
        Create a HE decryptor
        */
        void setDecryptor(); // decryptor
      
        /** 
        Create HE Relin keys for CSP
        */
        void setCSPHERelinKeys(); // csp_he_rk
       
        /** 
        Create HE Galois keys for CSP
        */
        void setCSPHEGaloisKeys();  // csp_he_gk
        
        /**
        Set up a data set name for NN calculation
        @param[in] data_set The data set name for NN calculation
        */
        void setDataSet(string data_set); // data_set


        // getter
        /**
        Return a HE key generator 
        */
        KeyGenerator* getKeyGenerator(); // analyst_keygen
       
        /** 
        Return a batch encoder
        */
        BatchEncoder* getBatchEncoder(); // analyst_he_benc
     
        /** 
        Return a HE Secret key
        */
        SecretKey getHESecretKey(); // analyst_he_sk
     
        /** 
        Return a HE Public key
        */
        PublicKey getHEPublicKey(); // analyst_he_pk
      
        /** 
        Return a HE Relin keys
        */
        RelinKeys getHERelinKeys(); // analyst_he_rk
      
        /** 
        Return a HE Galois keys
        */
        GaloisKeys getHEGaloisKeys(); // analyst_he_gk   
      
        /** 
        Return a HE encryptor
        */ 
        Encryptor* getEncryptor(); // analyst_he_enc
       
        /**
        Return a HE evaluator
        */
        Evaluator* getEvaluator(); // analyst_he_eval
      
        /**
        Return a HE decryptor
        */
        Decryptor* getDecryptor(); // analyst_he_dec
       
        /**
        Return the seal context
        */
        shared_ptr<SEALContext> getContext();
       
        /**
        Return a data set name for NN calculation
        */
        string getDataSet(); // data_set


        // functions
        /**
        Set up HE parameters
        */
        //void hEInitialization();
       
        /**
        Create HE keys
        */
        void generateHEKeys(const string& fileName);


        void saveHEKeys(const string& fileName);

        int loadHEKeys(const string& fileName);


        /**
        Return the byte size for HE Secret key
        */
        int getSecretKeyBytes(seal_byte* &);

       
        /**
        Return the byte size for HE Public key
        */
        int getPublicKeyBytes(seal_byte* &);
       
        /**
        Return the byte size for HE Relin keys
        */
        int getRelinKeyBytes(seal_byte* &);
       
        /**
        Return the byte size for HE Galois keys
        */
        int getGaloisKeyBytes(seal_byte* &);
      
        /**
        Return the byte size for HE Relin keys of CSP
        */
        int getCSPRelinKeyBytes(seal_byte* &);
       
        /**
        Return the byte size for HE Galois keys of CSP
        */
        int getCSPGaloisKeyBytes(seal_byte* &);
       
        /**
        Return the result for HE encryption of ML weights
        */
        vector<Ciphertext> getEncryptedWeights();
       
        /**
        Return the byte size for encrypted ML weights
        */
        int getEncryptedWeightsBytes(seal_byte* &, int index);
       
        /**
        Decrypt the Ciphertext from CSP and obtains the plaintext result
        */
        void decryptData(string patientId, seal_byte* bytes, int size);
       
        /** 
        Helper function to print the first ten bytes of the seal_byte input
        */
        void print_seal_bytes(seal_byte* buffer);

        void writePredictionsToFile(const string& patientId);

        // pure virtual function
        /**
        Create an abstract layer for NN model encryption
        */
        virtual void NNModelEncryption(string dataset) = 0;

    private:
        shared_ptr<SEALContext> context;

        PublicKey analyst_he_pk;   // Analyst HE Public key
        SecretKey analyst_he_sk;   // Analyst HE Secret key
        RelinKeys analyst_he_rk;   // Analyst HE Relin keys
        GaloisKeys analyst_he_gk;  // Analyst HE Galois keys
        RelinKeys csp_he_rk;       // CSP HE Relin keys
        GaloisKeys csp_he_gk;      // CSP HE Galois keys

        KeyGenerator* analyst_keygen;    // Analyst HE key generator
        BatchEncoder* analyst_he_benc;   // Analyst HE batch encoder
        Encryptor* analyst_he_enc;       // Analyst HE encryptor
        Evaluator* analyst_he_eval;      // Analyst HE evaluator
        Decryptor* analyst_he_dec;       // Analyst HE decryptor

        string dataset;   // The data set name for NN calculation

        // Map to store predictions associated with patientId
        std::map<std::string, std::vector<int64_t>> hhePredictions;

        // Mutex to protect access to hhePredictions
        std::mutex hhePredictions_mutex;

        void writeKeyToFile(ofstream& outFile,  seal_byte* keyBuffer, int keySize);
        void readKeyFromFile(ifstream& inFile,  seal_byte*& keyBuffer, int& keySize);

    protected:
        vector<Ciphertext> enc_weights_t; // The encrypted weight
        vector<int64_t> decrypted_result; // The decrypted result 
        int inputLen;
};

class Analyst_hhe_pktnn_1fc : public BaseAnalyst 
{
    public:
        /**
        The implementation of the pure virtual function 
        @param[in] dataset The data set name for ML
        */
        void NNModelEncryption(string dataset);
};