#include "hhe_pktnn_examples.h"

namespace hhe_pktnn_examples
{
    void print_vec_Ciphertext(std::vector<seal::Ciphertext> input, size_t size)
    {
        seal::seal_byte* buffer = nullptr;

        for (int i = 0; i < size; i++)
        {
            int input_size = input[i].save_size();
            buffer = new seal::seal_byte[input_size];
            input[i].save(buffer, input_size); 
            std::cout << "\n";
            for (int j=0; j < 10; j++){
                std::cout << (int)buffer[j] << " ";
            }

            cout << "input_size: " << input_size << endl;
        }
        std::cout << std::endl;
    }

    void print_Ciphertext(seal::Ciphertext input)
    {
        seal::seal_byte* buffer = nullptr;

        int input_size = input.save_size();
        buffer = new seal::seal_byte[input_size];
        input.save(buffer, input_size); 

        for (int j=0; j < 10; j++){
                std::cout << (int)buffer[j] << " ";
        }

        std::cout << std::endl;
    }

    void symmetric_key_he_encryption_test(vector<Ciphertext> enc_ssk,
                                          vector<uint64_t> ssk,
                                          bool USE_BATCH,
                                          std::shared_ptr<seal::SEALContext> context,
                                          const SecretKey &sk,
                                          const PublicKey &pk,
                                          const RelinKeys &rk,
                                          const GaloisKeys &gk,
                                          const BatchEncoder &he_benc,
                                          const Encryptor &he_enc)
    {
        pasta::PASTA_SEAL M1(context, pk, sk, rk, gk);
        auto enc_ssk_pasta = M1.encrypt_key_2(ssk, USE_BATCH);
        // vector<Ciphertext> enc_ssk = encrypt_symmetric_key(ssk, USE_BATCH, he_benc, he_enc);
        vector<uint64_t> dec_ssk_pasta = M1.decrypt_result(enc_ssk_pasta, USE_BATCH);
        vector<uint64_t> dec_ssk = M1.decrypt_result(enc_ssk, USE_BATCH);
        // cout << "dec_ssk.size = " << dec_ssk.size() << endl;
        // print_vec(dec_ssk_pasta, dec_ssk_pasta.size(), "dec_ssk_pasta");
        // print_vec(dec_ssk, dec_ssk.size(), "dec_ssk");
        if (dec_ssk != dec_ssk_pasta)
            throw runtime_error("decrypted symmetric keys are different");
        cout << "TEST: symmetric key encryption using HE test passed!" << endl;
    }

    int hhe_pktnn_ecg_inference()
    {
        utils::print_example_banner("PocketHHE: Privacy-preserving Inference with a 1-layer Neural Network on Integer ECG dataset");

        // the actors in the protocol
        Analyst analyst;
        Client client;
        CSP csp;

        // calculate the time (computation cost)
        std::chrono::high_resolution_clock::time_point analyst_start_0, analyst_end_0, analyst_start_1, analyst_end_1;
        std::chrono::high_resolution_clock::time_point client_start_0, client_end_0;
        std::chrono::high_resolution_clock::time_point csp_start_0, csp_end_0;
        std::chrono::milliseconds analyst_time_0, analyst_time_1, client_time_0, csp_time_0;

        // ---------------------- Analyst ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Analyst ----------------------"
                  << "\n";
        analyst_start_0 = std::chrono::high_resolution_clock::now(); // Start the timer

        std::cout << "Analyst constructs the HE context"
                  << "\n";
        std::shared_ptr<seal::SEALContext> context = sealhelper::get_seal_context(config::plain_mod, config::mod_degree, config::seclevel);
        sealhelper::print_parameters(*context);
        utils::print_line(__LINE__);
        std::cout << "Analyst creates the HE keys, batch encoder, encryptor and evaluator from the context"
                  << "\n";
        seal::KeyGenerator keygen(*context);
        analyst.he_sk = keygen.secret_key();     // HE secret key for decryption
        keygen.create_public_key(analyst.he_pk); // HE public key for encryption
        keygen.create_relin_keys(analyst.he_rk); // HE relinearization key to reduce noise in ciphertexts
        seal::BatchEncoder analyst_he_benc(*context);
        bool use_bsgs = false;
        std::vector<int> gk_indices = pastahelper::add_gk_indices(use_bsgs, analyst_he_benc);
        keygen.create_galois_keys(gk_indices, analyst.he_gk); // the HE Galois keys for batch computation
        seal::Encryptor analyst_he_enc(*context, analyst.he_pk);
        seal::Evaluator analyst_he_eval(*context);
        seal::Decryptor analyst_he_dec(*context, analyst.he_sk);

        utils::print_line(__LINE__);
        std::cout << "Analyst creates the neural network, loads the pretrained weights and biases"
                  << "\n";
        pktnn::pktfc fc(128, 1);
        if (config::debugging)
        {
            fc.loadWeight("../" + config::save_weight_path);
            fc.loadBias("../" + config::save_bias_path);
        }
        else
        {
            fc.loadWeight(config::save_weight_path);
            fc.loadBias(config::save_bias_path);
        }
        fc.printWeightShape();
        fc.printBiasShape();
        // get the weights and biases to encrypt later
        analyst.weight = fc.getWeight();
        analyst.bias = fc.getBias();
        // divide bias by 128
        int old_bias = analyst.bias.getElem(0, 0);
        analyst.bias.setElem(0, 0, (int)old_bias / 128);
        analyst.weight.printMat();
        analyst.bias.printMat();

        utils::print_line(__LINE__);
        std::cout << "Analyst encrypts the weights and biases using HE"
                  << "\n";
        pktnn::pktmat fc_weight_t;
        fc_weight_t.transposeOf(analyst.weight);
        analyst.enc_weight = sealhelper::encrypt_weight(fc_weight_t,
                                                        analyst.he_pk,
                                                        analyst_he_benc,
                                                        analyst_he_enc);
        std::cout << "Encrypt the weight...";
        std::cout << "The encrypted weight vector has " << analyst.enc_weight.size() << " ciphertexts\n";
        ecg_test::test_encrypted_weight(analyst.enc_weight,
                                        fc_weight_t,
                                        analyst.he_sk,
                                        analyst_he_benc,
                                        analyst_he_dec,
                                        128);
        std::cout << "Encrypt the bias...";
        analyst.enc_bias = sealhelper::encrypt_bias(analyst.bias,
                                                    analyst.he_pk,
                                                    analyst_he_enc);
        std::cout << "The encrypted bias vector has " << analyst.enc_bias.size() << " ciphertexts\n";
        ecg_test::test_encrypted_bias(analyst.enc_bias,
                                      analyst.bias,
                                      analyst.he_sk,
                                      analyst_he_dec);

        utils::print_line(__LINE__);
        std::cout << "Analyst sends the HE keys (except the secret key) to the CSP..."
                  << "\n";
        csp.he_gk = &analyst.he_gk;
        csp.he_pk = &analyst.he_pk;
        csp.he_rk = &analyst.he_rk;
        // calculate the commnication overhead (in MB)
        float he_pk_size = sealhelper::he_pk_key_size(analyst.he_pk, false);
        float he_keys_size = sealhelper::he_key_size(analyst.he_pk, analyst.he_rk, analyst.he_gk, true);

        utils::print_line(__LINE__);
        std::cout << "Analyst sends the encrypted weight and bias to the CSP..."
                  << "\n";
        csp.enc_weight = &analyst.enc_weight;
        csp.enc_bias = &analyst.enc_bias;
        // calculate the size of encrypted weights and biases (in MB)
        float enc_weight_bias_size = sealhelper::enc_weight_bias_size(analyst.enc_weight, analyst.enc_bias, true, true);
        analyst_end_0 = std::chrono::high_resolution_clock::now();
        analyst_time_0 = std::chrono::duration_cast<std::chrono::milliseconds>(analyst_end_0 - analyst_start_0);

        // ---------------------- Client (Data Owner) ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Client (Data Owner) ----------------------"
                  << "\n";
        client_start_0 = std::chrono::high_resolution_clock::now(); // Start the timer

        utils::print_line(__LINE__);
        std::cout << "Client loads his ECG test data" << std::endl;
        int numTestSamples = 13245;
        pktnn::pktmat ecgTestLabels(numTestSamples, 1);
        pktnn::pktmat ecgTestInput(numTestSamples, 128);
        pktnn::pktloader::loadTimeSeriesData(ecgTestInput, "data/mit-bih/csv/mitbih_x_test_int.csv",
                                             numTestSamples, config::debugging);

        pktnn::pktloader::loadTimeSeriesLabels(ecgTestLabels, "data/mit-bih/csv/mitbih_bin_y_test.csv",
                                               numTestSamples, config::debugging);
        ecgTestLabels.selfMulConst(128); // scale the output from 0-1 to 0-128

        if (config::dry_run) // get a slice of dry_run data samples
        {
            int dryRunNumSamples = config::dry_run_num_samples;
            std::cout << "Dry run: get a slice of " << dryRunNumSamples << " data samples"
                      << "\n";
            client.testData.sliceOf(ecgTestInput, 0, dryRunNumSamples - 1, 0, 127);
            client.testLabels.sliceOf(ecgTestLabels, 0, dryRunNumSamples - 1, 0, 0);
        }
        else
        {
            client.testData = ecgTestInput;
            client.testLabels = ecgTestLabels;
        }
        std::cout << "Test data shape: ";
        client.testData.printShape();
        // client.testData.printMat();
        std::cout << "Test labels shape: ";
        client.testLabels.printShape();

        utils::print_line(__LINE__);
        std::cout << "Client creates the symmetric key" << std::endl;
        client.k = pastahelper::get_symmetric_key();
        std::cout << "Symmetric key size: " << client.k.size() << "\n";
        // utils::print_vec(client.k, client.k.size(), "Symmetric key: ");

        utils::print_line(__LINE__);
        std::cout << "Client encrypts his symmetric key using HE" << std::endl;
        client.c_k = pastahelper::encrypt_symmetric_key(client.k, config::USE_BATCH, analyst_he_benc, analyst_he_enc);

        utils::print_line(__LINE__);
        std::cout << "Client symmetrically encrypts his ECG data" << std::endl;
        pasta::PASTA SymmetricEncryptor(client.k, config::plain_mod);
        client.cs = pastahelper::symmetric_encrypt(SymmetricEncryptor, client.testData); // the symmetric encrypted images
        std::cout << "The symmetric encrypted data has " << client.cs.size() << " ciphertexts\n";

        utils::print_line(__LINE__);
        std::cout << "The client sends the symmetric encrypted data and the HE encrypted symmetric key to the CSP..."
                  << "\n";
        csp.c_k = &client.c_k;
        csp.cs = &client.cs;
        // calculate the size of the symmetric encrypted data and HE encrypted symmetric key (in MB)
        float sym_enc_data_size = pastahelper::sym_enc_data_size(client.cs, true);
        float he_enc_sym_key_size = sealhelper::he_vec_size(client.c_k, true, "HE encrypted symmetric key");
        client_end_0 = std::chrono::high_resolution_clock::now();
        client_time_0 = std::chrono::duration_cast<std::chrono::milliseconds>(client_end_0 - client_start_0);

        // -------------------------- CSP (server) ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "-------------------------- CSP ----------------------" << std::endl;
        csp_start_0 = std::chrono::high_resolution_clock::now(); // Start the timer

        utils::print_line(__LINE__);
        std::cout << "CSP runs the decomposition algorithm to turn the symmetric encrypted data into HE encrypted data" << std::endl;
        seal::KeyGenerator csp_keygen(*context);
        csp.he_sk = csp_keygen.secret_key();
        // Below is to check if the csp key is different to the analyst key (they must be different)
        // csp.he_sk.save(std::cout);
        // std::cout << "\n";
        // analyst.he_sk.save(std::cout);
        // std::cout << "\n";
        pasta::PASTA_SEAL HHE(context, *csp.he_pk, csp.he_sk, *csp.he_rk, *csp.he_gk);
        for (std::vector<uint64_t> c : *csp.cs)
        {
            std::vector<seal::Ciphertext> c_prime = HHE.decomposition(c, *csp.c_k, config::USE_BATCH);
            if (c_prime.size() == 1)
            {
                csp.c_primes.push_back(c_prime[0]);

                // --- for debugging: we decrypt the decomposed ciphertexts with the analyst's secret key
                // to check if the decryption is same as the plaintext data of the client
                // std::vector<int64_t> dec_c_prime = sealhelper::decrypting(c_prime[0], analyst.he_sk, analyst_he_benc, *context, 128);
                // utils::print_vec(dec_c_prime, dec_c_prime.size(), "decrypted c_prime ", "\n");
            }
            else
            {
                std::cout << "there are more than 1 seal ciphertexts in the each decomposed ciphertext\n";
                std::cout << "we need to do some post-processing\n";
            }
        }
        std::cout << "There are " << csp.c_primes.size() << " decomposed HE ciphertexts\n";

        utils::print_line(__LINE__);
        std::cout << "CSP evaluates the HE encrypted weights (& biases) on the HE encrypted data" << std::endl;
        for (seal::Ciphertext c_prime : csp.c_primes)
        {
            seal::Ciphertext enc_result;
            // std::vector<seal::Ciphertext> csp_enc_weight = *csp.enc_weight;
            sealhelper::packed_enc_multiply(c_prime, (*csp.enc_weight)[0],
                                            enc_result, analyst_he_eval);
            // we only do element-wise multiplication for now and ignore the
            // bias for simplication as it does not affect the result
            csp.enc_results.push_back(enc_result);
        }

        utils::print_line(__LINE__);
        std::cout << "CSP sends the HE encrypted result to the analyst" << std::endl;
        analyst.enc_results = &csp.enc_results;
        float enc_results_size = sealhelper::he_vec_size(csp.enc_results, true, "HE encrypted results");
        csp_end_0 = std::chrono::high_resolution_clock::now();
        csp_time_0 = std::chrono::duration_cast<std::chrono::milliseconds>(csp_end_0 - csp_start_0);

        // ---------------------- Analyst (again) ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Analyst ----------------------"
                  << "\n";
        analyst_start_1 = std::chrono::high_resolution_clock::now();

        utils::print_line(__LINE__);
        std::cout << "The analyst decrypts the HE encrypted results received from the CSP" << std::endl;
        for (seal::Ciphertext enc_result : *analyst.enc_results)
        {
            std::vector<int64_t> dec_result = sealhelper::decrypting(enc_result,
                                                                     analyst.he_sk,
                                                                     analyst_he_benc,
                                                                     *context,
                                                                     128);
            analyst.dec_results.push_back(dec_result);
            // utils::print_vec(dec_result, dec_result.size(), "decrypted result ", ", ");
        }

        utils::print_line(__LINE__);
        std::cout << "The analyst applies the non-linear operations on the decrypted results and get the final predictions" << std::endl;
        for (std::vector<int64_t> dec_result : analyst.dec_results)
        {
            // first find the sum of the decrypted results
            int sum = 0;
            for (auto i : dec_result)
            {
                sum += i;
            }
            // std::cout << "sum = " << sum << "\n";
            // apply the pocket sigmoid function
            int out = utils::simple_pocket_sigmoid(sum);
            // the final prediction
            int final_pred = 0;
            out > 64 ? final_pred = 128 : final_pred = 0;
            // add the prediction to the analyst's predictions
            analyst.predictions.push_back(final_pred);
        }

        // find the accuracy
        int testNumCorrect = 0;
        for (int i = 0; i < analyst.predictions.size(); ++i)
        {
            if (config::verbose)
                std::cout << "Prediction = " << analyst.predictions[i]
                          << "| Actual = " << client.testLabels.getElem(i, 0) << "\n";
            if (client.testLabels.getElem(i, 0) == analyst.predictions[i])
            {
                ++testNumCorrect;
            }
        }
        analyst_end_1 = std::chrono::high_resolution_clock::now();
        analyst_time_1 = std::chrono::duration_cast<std::chrono::milliseconds>(analyst_end_1 - analyst_start_1);

        // ---------------------- Experiment results ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Experiment Results ----------------------"
                  << "\n";
        std::cout << "Final correct predions = " << testNumCorrect << " (out of "
                  << analyst.predictions.size() << " total examples)"
                  << "\n";
        std::cout << "Encrypted accuracy = "
                  << (double)testNumCorrect / analyst.predictions.size() * 100 << "% \n \n";
        // print out communication and computation costs here
        utils::print_line(__LINE__);
        std::cout << "Computation cost: " << std::endl;
        size_t analyst_time_ms = analyst_time_0.count() + analyst_time_1.count();
        size_t total_time = client_time_0.count() + analyst_time_ms + csp_time_0.count();
        utils::print_time("Analyst", analyst_time_ms);
        utils::print_time("Client", client_time_0.count());
        utils::print_time("CSP", csp_time_0.count());
        utils::print_time("Total", total_time);
        std::cout << "\n";

        utils::print_line(__LINE__);
        std::cout << "Communication cost: " << std::endl;
        std::cout << "Analyst - Client : " << he_pk_size << " (Mb)" << std::endl;
        std::cout << "Client - CSP: " << sym_enc_data_size + he_enc_sym_key_size << " (Mb)" << std::endl;
        std::cout << "Analyst - CSP: " << he_keys_size + enc_weight_bias_size + enc_results_size << " (Mb)" << std::endl;
        float total_comm = sym_enc_data_size + he_enc_sym_key_size + he_pk_size +
                           he_keys_size + enc_weight_bias_size + enc_results_size;
        std::cout << "Total communication cost: " << total_comm << " (Mb)" << std::endl;

        return 0;
    }

    int hhe_pktnn_1fc_inference(const std::string &dataset)
    {
        utils::print_example_banner("HHE Inference with a 1-FC Neural Network");
        std::cout << "Dataset: " << dataset << std::endl;

        // check if the lowercase of the `dataset` string is either "spo2" or "mnist"
        std::string lowerStr = dataset;
        std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
        if (lowerStr != "spo2" && lowerStr != "ecg")
        {
            throw std::runtime_error("Dataset must be either SpO2 or ECG");
        }
        int inputLen = 0;
        if (lowerStr == "spo2")
        {
            inputLen = 300;
        }
        if (lowerStr == "ecg")
        {
            inputLen = 128;
        }

        // the actors in the protocol
        Analyst analyst;
        Client client;
        CSP csp;

        // calculate the time (computation cost)
        std::chrono::high_resolution_clock::time_point time_start, time_end;
        std::chrono::milliseconds time_diff;

        std::chrono::high_resolution_clock::time_point analyst_start_0, analyst_end_0, analyst_start_1, analyst_end_1;
        std::chrono::high_resolution_clock::time_point client_start_0, client_end_0;
        std::chrono::high_resolution_clock::time_point csp_start_0, csp_end_0;
        std::chrono::milliseconds analyst_time_0, analyst_time_1, client_time_0, csp_time_0;

        // ---------------------- Analyst ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Analyst ----------------------"
                  << "\n";
        analyst_start_0 = std::chrono::high_resolution_clock::now(); // Start the timer

        std::cout << "Analyst constructs the HE context"
                  << "\n";
        std::shared_ptr<seal::SEALContext> context = sealhelper::get_seal_context(config::plain_mod, config::mod_degree, config::seclevel);
        sealhelper::print_parameters(*context);
        utils::print_line(__LINE__);
        std::cout << "Analyst creates the HE keys, batch encoder, encryptor and evaluator from the context"
                  << "\n";
        seal::KeyGenerator keygen(*context);
        analyst.he_sk = keygen.secret_key();     // HE secret key for decryption
        keygen.create_public_key(analyst.he_pk); // HE public key for encryption
        keygen.create_relin_keys(analyst.he_rk); // HE relinearization key to reduce noise in ciphertexts
        seal::BatchEncoder analyst_he_benc(*context);
        bool use_bsgs = false;
        std::vector<int> gk_indices = pastahelper::add_gk_indices(use_bsgs, analyst_he_benc);
        // keygen.create_galois_keys(gk_indices, analyst.he_gk); // the HE Galois keys for batch computation
        keygen.create_galois_keys(analyst.he_gk); // the HE Galois keys for batch computation
        seal::Encryptor analyst_he_enc(*context, analyst.he_pk);
        seal::Evaluator analyst_he_eval(*context);
        seal::Decryptor analyst_he_dec(*context, analyst.he_sk);
        size_t slot_count = analyst_he_benc.slot_count();
        std::cout << "Batch encoder slot count = " << slot_count << std::endl;

        cout << "analyst_he_pk size: " << (analyst.he_pk).save_size() << endl;
        cout << "analyst_he_rk size: " << (analyst.he_rk).save_size() << endl;
        cout << "analyst_he_gk size: " << (analyst.he_gk).save_size() << endl;
        cout << "analyst_he_sk size: " << (analyst.he_sk).save_size() << endl;



        utils::print_line(__LINE__);
        std::cout << "Analyst loads the pretrained weights"
                  << "\n";
        matrix::matrix weights;
        if (config::debugging)
        {
            weights = matrix::read_from_csv("../" + config::save_weight_path);
        }
        else
        {
            weights = matrix::read_from_csv(config::save_weight_path);
        }
        std::cout << "Reading weights from " << config::save_weight_path << std::endl;
        matrix::print_matrix_shape(weights);
        matrix::print_matrix_stats(weights);
        // matrix::print_matrix(weights);
        std::cout << "Transposed weights: ";
        matrix::matrix weights_t = matrix::transpose(weights);
        matrix::print_matrix_shape(weights_t);
        matrix::print_matrix(weights_t);

        // bias (all 0s)
        std::cout << "Ignoring Bias" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "Analyst encrypts the weights using HE" << std::endl;
        std::vector<seal::Ciphertext> enc_weights_t = sealhelper::encrypt_weight_mat(weights_t,
                                                                                     analyst.he_pk,
                                                                                     analyst_he_benc,
                                                                                     analyst_he_enc);
        std::cout << "print encrypted weights size: " << std::endl;
        print_vec_Ciphertext(enc_weights_t, enc_weights_t.size());
        cout << "enc_weights_t size: " << enc_weights_t.size() << endl;

        utils::print_line(__LINE__);
        std::cout << "(Check) Analyst decrypts the encrypted weight" << std::endl;
        matrix::matrix dec_weights_t = sealhelper::decrypt_weight_mat(enc_weights_t,
                                                                      analyst_he_benc,
                                                                      analyst_he_dec,
                                                                      inputLen);
        std::cout << "Decrypted Weights: ";
        matrix::print_matrix_shape(dec_weights_t);
        matrix::print_matrix(dec_weights_t);

        // ---------------------- Client (Data Owner) ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Client (Data Owner) ----------------------"
                  << std::endl;
        client_start_0 = std::chrono::high_resolution_clock::now(); // Start the timer
        utils::print_line(__LINE__);
        std::cout << "Client loads his input data from " << config::dataset_input_path << std::endl;
        matrix::matrix data = matrix::read_from_csv(config::dataset_input_path);
        // matrix::print_matrix(data);
        matrix::print_matrix_shape(data);
        matrix::print_matrix_stats(data);
        std::cout << "Client loads his labels data from " << config::dataset_output_path << std::endl;
        matrix::matrix labels = matrix::read_from_csv(config::dataset_output_path);
        // matrix::print_matrix(labels);
        matrix::print_matrix_shape(labels);
        matrix::print_matrix_stats(labels);

        utils::print_line(__LINE__);
        std::cout << "(Check) Computing in plain on 1 input vector" << std::endl;
        matrix::vector vo_p(1);
        size_t data_index = 5;
        matrix::vector vi = data[data_index];
        int64_t gt_out = labels[data_index][0];
        std::cout << "input vector vi.size() = " << vi.size() << ";\n";
        utils::print_vec(vi, vi.size(), "vi");
        matrix::matMulVecNoModulus(vo_p, weights_t, vi);
        std::cout << "plain output vector vo.size() = " << vo_p.size() << ";\n";
        utils::print_vec(vo_p, vo_p.size(), "vo_p");
        int64_t plain_pred = utils::int_sigmoid(vo_p[0]);
        std::cout << "plain prediction = " << plain_pred << " | ";
        std::cout << "groundtruth label = " << gt_out << ";\n";

        utils::print_line(__LINE__);
        std::cout << "Client symmetrically encrypts input" << std::endl;
        std::vector<uint64_t> client_sym_key = pastahelper::get_symmetric_key();
        pasta::PASTA SymmetricEncryptor(client_sym_key, config::plain_mod);
        std::vector<uint64_t> vi_se = pastahelper::symmetric_encrypt_vec(SymmetricEncryptor, vi); // the symmetric encrypted images
        utils::print_vec(vi_se, vi_se.size(), "vi_se");
        cout << "encrypted data size: " << vi_se.size() << endl;

        std::cout << "(Check) Client decrypts symmetrically encrypted input" << std::endl;
        std::vector<uint64_t> vi_dec = pastahelper::symmetric_decrypt_vec(SymmetricEncryptor, vi_se); // the symmetric encrypted images
        utils::print_vec(vi_dec, vi_dec.size(), "vi_dec");

        utils::print_line(__LINE__);
        seal::BatchEncoder analyst_he_benc_1111(*context);
        std::cout << "Client encrypts the symmetric key using HE (the HHE key)" << std::endl;
        std::vector<seal::Ciphertext> client_hhe_key = pastahelper::encrypt_symmetric_key(
            client_sym_key, config::USE_BATCH, analyst_he_benc_1111, analyst_he_enc);

        std::cout << "print client_hhe key: " << std::endl;
        print_vec_Ciphertext(client_hhe_key, client_hhe_key.size());
        cout << "client_hhe_key size: " << client_hhe_key.size() << endl; 

        seal::BatchEncoder analyst_he_benc_2222(*context);
        std::cout << "client_hhe_key test: " << std::endl;   
        symmetric_key_he_encryption_test(client_hhe_key, 
                                         client_sym_key, 
                                         config::USE_BATCH, 
                                         context,
                                         analyst.he_sk, 
                                         analyst.he_pk, 
                                         analyst.he_rk, 
                                         analyst.he_gk,
                                         analyst_he_benc_2222, 
                                         analyst_he_enc);

        // -------------------------- CSP ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "-------------------------- CSP ----------------------" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "CSP does HHE decomposition to turn client's symmetric input into HE input\n";
        seal::KeyGenerator csp_keygen(*context); // CSP creates a new sk key for himself
        csp.he_sk = csp_keygen.secret_key();
        pasta::PASTA_SEAL HHE(context, analyst.he_pk, csp.he_sk, analyst.he_rk, analyst.he_gk);
        time_start = std::chrono::high_resolution_clock::now();
        std::vector<seal::Ciphertext> vi_he = HHE.decomposition(vi_se, client_hhe_key, config::USE_BATCH);
        std::cout<< "print vi_he: " << std::endl;
        print_vec_Ciphertext(vi_he, vi_he.size());
        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            time_end - time_start);
        std::cout << "Time: " << time_diff.count() << " milliseconds"
                  << " = " << time_diff.count() / 1000 << " seconds" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "CSP does HHE decomposition postprocessing on the HE encrypted input" << std::endl;
        size_t num_block = inputLen / HHE.get_plain_size();
        size_t rem = inputLen % HHE.get_plain_size();
        if (rem)
        {
            num_block++;
        }
        std::cout << "There are " << vi_he.size() << " decomposed HE ciphertexts\n";
        std::cout << "HHE cipher one block's plain size " << HHE.get_plain_size() << std::endl;
        std::cout << "num_block = " << num_block << std::endl;
        std::cout << "rem = " << rem << std::endl;
        std::cout << "Preparing necessary things to do postprocessing (creating new Galois key, masking, flattening)" << std::endl;
        std::vector<int> flatten_gks;
        for (int i = 1; i < num_block; i++)
        {
            flatten_gks.push_back(-(int)(i * HHE.get_plain_size()));
        }

        utils::print_vec(gk_indices, gk_indices.size(), "gk_indices");
        utils::print_vec(flatten_gks, flatten_gks.size(), "flatten_gks");

        std::vector<int> csp_gk_indices = pastahelper::add_some_gk_indices(gk_indices, flatten_gks);
        utils::print_vec(csp_gk_indices, csp_gk_indices.size(), "csp_gk_indices");
        
        seal::GaloisKeys csp_gk;
        keygen.create_galois_keys(csp_gk_indices, csp_gk);
        seal::RelinKeys csp_rk;
        keygen.create_relin_keys(csp_rk);

        time_start = std::chrono::high_resolution_clock::now();
        if (rem != 0)
        {
            std::vector<uint64_t> mask(rem, 1);
            HHE.mask(vi_he.back(), mask);
        }
        seal::Ciphertext vi_he_processed;
        HHE.flatten(vi_he, vi_he_processed, csp_gk);
        std::cout<< "print vi_he_processed: " << std::endl;
        print_Ciphertext(vi_he_processed);

        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            time_end - time_start);
        std::cout << "Time: " << time_diff.count() << " milliseconds"
                  << " = " << time_diff.count() / 1000 << " seconds" << std::endl;

        utils::print_line(__LINE__);
        seal::BatchEncoder analyst_he_benc2(*context);
        std::cout << "(Check) Decrypts processed, decomposed HE input vector using Analyst's HE secret key\n";
        std::vector<int64_t> vi_he_processed_decrypted = sealhelper::decrypting(vi_he_processed,
                                                                                analyst.he_sk,
                                                                                analyst_he_benc2,
                                                                                *context,
                                                                                inputLen);
        utils::print_vec(vi_he_processed_decrypted, vi_he_processed_decrypted.size(), "vi_he_decrypted_processed");
        if (vi_he_processed_decrypted.size() != vi.size())
        {
            throw std::logic_error("The decrypted HE input vector after decomposition has different length than the plaintext version!");
        }

        utils::print_line(__LINE__);
        std::cout << "CSP evaluates the HE weights on the decomposed HE data" << std::endl;
        seal::Ciphertext encrypted_product;
        sealhelper::packed_enc_multiply(vi_he_processed, enc_weights_t[0],
                                        encrypted_product, analyst_he_eval);

        std::cout << "encrypted_product size before relinearization = " << encrypted_product.size() << std::endl;
        analyst_he_eval.relinearize_inplace(encrypted_product, csp_rk);
        std::cout << "encrypted_product size after relinearization = " << encrypted_product.size() << std::endl;
        utils::print_line(__LINE__);
        std::cout << "(Check) Decrypt the encrypted product to check" << std::endl;
        std::vector<int64_t> decrypted_product = sealhelper::decrypting(encrypted_product,
                                                                        analyst.he_sk,
                                                                        analyst_he_benc,
                                                                        *context,
                                                                        inputLen);
        utils::print_vec(decrypted_product, decrypted_product.size(), "decrypted_product");

        // Do encrypted sum on the resulting product vector
        utils::print_line(__LINE__);
        std::cout << "CSP does encrypted sum on the encrypted vector" << std::endl;
        seal::Ciphertext encrypted_sum_vec;
        sealhelper::encrypted_vec_sum(
            encrypted_product, encrypted_sum_vec, analyst_he_eval, analyst.he_gk, inputLen);

        std::cout
            << "\n---------------------- Analyst ----------------------"
            << "\n";
        utils::print_line(__LINE__);
        std::cout << "Analyst decrypts the HE encrypted results received from the CSP" << std::endl;
        std::vector<int64_t> decrypted_result = sealhelper::decrypting(encrypted_sum_vec,
                                                                       analyst.he_sk,
                                                                       analyst_he_benc,
                                                                       *context,
                                                                       inputLen);
        utils::print_vec(decrypted_result, decrypted_result.size(), "decrypted sum vector");
        matrix::vector vo(1);
        vo[0] = decrypted_result[inputLen - 1];

        std::cout << "Plaintext FC layer output: " << vo_p[0] << std::endl;
        std::cout << "Decrypted HHE FC layer output: " << vo[0] << std::endl;

        if (vo != vo_p)
        {
            utils::print_line(__LINE__);
            std::cout << "!!!HHE Protocol Failed!!!\n";
            utils::print_vec(vo_p, vo_p.size(), "Plaintext result");
            utils::print_vec(vo, vo.size(), "HHE result");
            throw std::runtime_error("FC layer's plaintext results and HHE results are different");
        }

        utils::print_line(__LINE__);
        std::cout << "Analyst applies the sigmoid to get final prediction" << std::endl;
        int64_t hhe_pred = utils::int_sigmoid(vo[0]);
        std::cout << "HHE prediction = " << hhe_pred << " | ";
        std::cout << "plain prediction = " << plain_pred << " | ";
        std::cout << "ground-truth prediction = " << gt_out << std::endl;

        std::cout << "\n---------------------- Done ----------------------"
                  << "\n";
        return 0;
    }

    int hhe_pktnn_2fc_inference(const std::string &dataset)
    {
        utils::print_example_banner("HHE Inference with a 2-FC Neural Network & Square Activation Function");
        std::cout << "Dataset: " << dataset << std::endl;

        std::string lowerStr = dataset;
        std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
        if (lowerStr != "mnist" && lowerStr != "fmnist")
        {
            throw std::runtime_error("Dataset must be either MNIST or FMNIST");
        }
        int inputLen = 784;

        // the actors in the protocol
        Analyst analyst;
        Client client;
        CSP csp;

        // Some time variables to calculate operations' runtimes
        std::chrono::high_resolution_clock::time_point time_start;
        std::chrono::high_resolution_clock::time_point time_end;
        std::chrono::milliseconds time_diff;

        // ---------------------- Analyst ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Analyst (Model Owner) ----------------------"
                  << "\n";

        std::cout << "Analyst constructs the HE context"
                  << "\n";
        std::shared_ptr<seal::SEALContext> context = sealhelper::get_seal_context(
            config::plain_mod, config::mod_degree, config::seclevel);
        sealhelper::print_parameters(*context);
        utils::print_line(__LINE__);
        std::cout << "Analyst creates the HE keys, batch encoder, encryptor and evaluator from the context"
                  << "\n";
        seal::KeyGenerator keygen(*context);
        analyst.he_sk = keygen.secret_key();     // HE secret key for decryption
        keygen.create_public_key(analyst.he_pk); // HE public key for encryption
        keygen.create_relin_keys(analyst.he_rk); // HE relinearization key to reduce noise in ciphertexts
        seal::BatchEncoder analyst_he_benc(*context);
        bool use_bsgs = false;
        std::vector<int> gk_indices = pastahelper::add_gk_indices(use_bsgs, analyst_he_benc);
        keygen.create_galois_keys(analyst.he_gk); // the HE Galois keys for batch computation
        seal::Encryptor analyst_he_enc(*context, analyst.he_pk);
        seal::Evaluator analyst_he_eval(*context);
        seal::Decryptor analyst_he_dec(*context, analyst.he_sk);
        size_t slot_count = analyst_he_benc.slot_count();
        std::cout << "Batch encoder slot count = " << slot_count << std::endl;

        utils::print_line(__LINE__);
        std::cout << "Analyst loads the pretrained weights"
                  << "\n";
        matrix::matrix fc1;
        matrix::matrix fc2;
        std::string fc1_path;
        std::string fc2_path;
        if (config::debugging)
        {
            fc1_path = "../" + config::save_weight_path + "_fc1.csv";
            fc2_path = "../" + config::save_weight_path + "_fc2.csv";
        }
        else
        {
            fc1_path = config::save_weight_path + "_fc1.csv";
            fc2_path = config::save_weight_path + "_fc2.csv";
        }
        std::cout << "Reading fc1 from " << fc1_path << std::endl;
        fc1 = matrix::read_from_csv(fc1_path);
        matrix::matrix fc1_t = matrix::transpose(fc1);
        matrix::print_matrix_shape(fc1_t, "fc1_t");
        matrix::print_matrix_stats(fc1_t);

        std::cout << "Reading fc2 from " << fc2_path << std::endl;
        fc2 = matrix::read_from_csv(fc2_path);
        matrix::matrix fc2_t = matrix::transpose(fc2);
        matrix::print_matrix_shape(fc2_t, "fc2_t");
        matrix::print_matrix_stats(fc2_t);

        std::cout << "Ignoring Bias" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "Analyst encrypts the fc1 using HE" << std::endl;
        std::vector<seal::Ciphertext> he_fc1_t = sealhelper::encrypt_weight_mat(fc1_t,
                                                                                analyst.he_pk,
                                                                                analyst_he_benc,
                                                                                analyst_he_enc);
        std::cout << "he_fc1_t has " << he_fc1_t.size() << " HE ciphertexts" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "(Check) Analyst decrypts the HE encrypted fc1" << std::endl;
        matrix::matrix dec_fc1_t = sealhelper::decrypt_weight_mat(he_fc1_t,
                                                                  analyst_he_benc,
                                                                  analyst_he_dec,
                                                                  inputLen);
        matrix::print_matrix_shape(dec_fc1_t, "Decrypted fc1_t");
        // matrix::print_matrix(dec_fc1_1);
        checks::are_same_matrices(fc1_t, dec_fc1_t, "fc1_t", "dec_fc1_t");

        utils::print_line(__LINE__);
        std::cout << "TODO: Analyst encrypts the fc2 using HE" << std::endl;
        std::vector<std::vector<seal::Ciphertext>> he_fc2_t = sealhelper::encrypt_weight_mat_no_batch(fc2_t,
                                                                                                      analyst.he_pk,
                                                                                                      analyst_he_benc,
                                                                                                      analyst_he_enc);

        // ---------------------- Client (Data Owner) ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "---------------------- Client (Data Owner) ----------------------"
                  << std::endl;
        utils::print_line(__LINE__);
        std::cout << "Client loads his input data from " << config::dataset_input_path << std::endl;
        matrix::matrix data = matrix::read_from_csv(config::dataset_input_path);
        // matrix::print_matrix(data);
        matrix::print_matrix_shape(data);
        matrix::print_matrix_stats(data);
        std::cout << "Client loads his labels data from " << config::dataset_output_path << std::endl;
        matrix::matrix labels = matrix::read_from_csv(config::dataset_output_path);
        // matrix::print_matrix(labels);
        matrix::print_matrix_shape(labels);
        matrix::print_matrix_stats(labels);

        utils::print_line(__LINE__);
        std::cout << "(Check) Computing in plain on 1 input vector" << std::endl;
        size_t data_index = 0;
        matrix::vector vi = data[data_index];
        int64_t gt_out = labels[data_index][0];
        std::cout << "input vector vi.size() = " << vi.size() << ";\n";
        std::cout << "-- 1st linear layer: ";
        matrix::vector vo_p1;
        matrix::matMulVecNoModulus(vo_p1, fc1_t, vi);
        std::cout << "vo_p1.size() = " << vo_p1.size() << ";\n";
        utils::print_vec(vo_p1, vo_p1.size(), "vo_p1", ", ");
        std::cout << "-- square activation: ";
        matrix::vector vo_p_square;
        matrix::square(vo_p_square, vo_p1);
        std::cout << "vo_p_square.size() = " << vo_p_square.size() << ";\n";
        utils::print_vec(vo_p_square, vo_p_square.size(), "vo_p_square", ", ");
        std::cout << "-- 2nd linear layer: ";
        matrix::vector vo_p2;
        matrix::matMulVecNoModulus(vo_p2, fc2_t, vo_p_square);
        std::cout << "vo_p2.size() = " << vo_p2.size() << ";\n";
        utils::print_vec(vo_p2, vo_p2.size(), "vo_p2");
        int pred_plain = matrix::argmax(vo_p2);
        std::cout << "-- plain prediction = " << pred_plain << " | "
                  << "groundtruth label = " << gt_out << ";\n";
        !(pred_plain == gt_out) ? throw std::runtime_error("Assertion failed: plain prediction and ground truth are not the same")
                                : std::cout << "Check pass: Correct prediction" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "Client symmetrically encrypts input" << std::endl;
        std::vector<uint64_t> client_sym_key = pastahelper::get_symmetric_key();
        pasta::PASTA SymmetricEncryptor(client_sym_key, config::plain_mod);
        std::vector<uint64_t> vi_se = pastahelper::symmetric_encrypt_vec(SymmetricEncryptor, vi); // the symmetric encrypted images
        std::cout << "vi_se.size() = " << vi_se.size() << std::endl;
        utils::print_vec(vi_se, 10, "vi_se (first 10 values)");

        utils::print_line(__LINE__);
        std::cout << "(Check) Client decrypts the symmetrically encrypted input" << std::endl;
        std::vector<uint64_t> vi_dec = pastahelper::symmetric_decrypt_vec(SymmetricEncryptor, vi_se); // the symmetric encrypted images
        // utils::print_vec(vi_dec, vi_dec.size(), "vi_dec");
        checks::are_same_vectors(vi, vi_dec);

        utils::print_line(__LINE__);
        std::cout << "Client encrypts the symmetric key using HE (the HHE key)" << std::endl;
        std::vector<seal::Ciphertext> client_hhe_key = pastahelper::encrypt_symmetric_key(
            client_sym_key, config::USE_BATCH, analyst_he_benc, analyst_he_enc);

        // -------------------------- CSP ----------------------
        std::cout << "\n";
        utils::print_line(__LINE__);
        std::cout << "-------------------------- CSP ----------------------" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "CSP creates his own HE secret key\n";
        seal::KeyGenerator csp_keygen(*context);
        csp.he_sk = csp_keygen.secret_key();
        checks::are_same_he_sk(csp.he_sk, analyst.he_sk);

        utils::print_line(__LINE__);
        std::cout << "CSP does HHE decomposition to turn client's symmetric input into HE input...\n";
        pasta::PASTA_SEAL HHE(context, analyst.he_pk, csp.he_sk, analyst.he_rk, analyst.he_gk);
        time_start = std::chrono::high_resolution_clock::now();
        std::vector<seal::Ciphertext> vi_he = HHE.decomposition(vi_se, client_hhe_key, config::USE_BATCH);
        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            time_end - time_start);
        std::cout << "Decomposition Time: " << time_diff.count() << " milliseconds"
                  << " = " << time_diff.count() / 1000 << " seconds" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "CSP does decomposition postprocessing on the HE encrypted input..." << std::endl;
        size_t num_block = inputLen / HHE.get_plain_size();
        size_t rem = inputLen % HHE.get_plain_size();
        if (rem)
            num_block++;
        std::cout << "There are " << vi_he.size() << " decomposed HE ciphertexts. The reason is: " << std::endl;
        std::cout << "- HHE cipher one block's plain size " << HHE.get_plain_size() << std::endl;
        std::cout << "- (number_of_blocks, remaining) = " << inputLen << "/"
                  << HHE.get_plain_size() << " = (" << num_block << ", " << rem << ")\n";

        utils::print_line(__LINE__);
        std::cout << "Preparing necessary things to do decomposition postprocessing "
                  << "(new Galois key, masking, flattening) " << std::endl;
        std::vector<int> flatten_gks;
        for (int i = 1; i < num_block; i++)
            flatten_gks.push_back(-(int)(i * HHE.get_plain_size()));
        std::vector<int> csp_gk_indices = pastahelper::add_some_gk_indices(gk_indices, flatten_gks);
        seal::GaloisKeys csp_gk;
        keygen.create_galois_keys(csp_gk_indices, csp_gk);
        seal::RelinKeys csp_rk;
        keygen.create_relin_keys(csp_rk);

        time_start = std::chrono::high_resolution_clock::now();
        if (rem != 0)
        {
            std::vector<uint64_t> mask(rem, 1);
            HHE.mask(vi_he.back(), mask);
        }
        seal::Ciphertext vi_he_processed;
        HHE.flatten(vi_he, vi_he_processed, csp_gk);
        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            time_end - time_start);
        std::cout << "Time: " << time_diff.count() << " milliseconds"
                  << " = " << time_diff.count() / 1000 << " seconds" << std::endl;

        utils::print_line(__LINE__);
        std::cout << "(Check) Decrypts the processed, decomposed HE input vector using "
                  << "Analyst's HE secret key and compare with the plaintext input vector\n";
        std::vector<int64_t> vi_he_processed_decrypted = sealhelper::decrypting(vi_he_processed,
                                                                                analyst.he_sk,
                                                                                analyst_he_benc,
                                                                                *context,
                                                                                inputLen);
        utils::print_vec(vi_he_processed_decrypted, 10, "vi_he_decrypted_processed (first 10 values)");
        checks::are_same_vectors(vi, vi_he_processed_decrypted);

        utils::print_line(__LINE__);
        std::cout << "CSP evaluates the HE fc1 on the decomposed HE data..." << std::endl;
        std::vector<seal::Ciphertext> vo_he1;
        size_t i = 0;
        for (const seal::Ciphertext &he_fc1_row : he_fc1_t)
        {
            seal::Ciphertext encrypted_elem_product;
            utils::print_line(__LINE__);
            std::cout << "Doing encrypted element-wise vector multiplication:" << std::endl;
            sealhelper::packed_enc_multiply(vi_he_processed, he_fc1_row,
                                            encrypted_elem_product, analyst_he_eval);
            // relinearizes ciphertext and reduce its ciphertext size to 2 (from 3)
            analyst_he_eval.relinearize_inplace(encrypted_elem_product, csp_rk);
            // std::cout << encrypted_elem_product.size() << std::endl;
            // utils::print_vec(decrypted_elem_product, decrypted_elem_product.size(), "decrypted_elem_product");
            // Do encrypted sum on the resulting element-wise product vector
            utils::print_line(__LINE__);
            std::cout << "CSP does encrypted sum on the encrypted vector" << std::endl;
            seal::Ciphertext encrypted_sum_vec;
            sealhelper::encrypted_vec_sum(encrypted_elem_product, encrypted_sum_vec,
                                          analyst_he_eval, analyst.he_gk, inputLen);
            std::cout << "(Check) Decrypt the encrypted sum vector to check. "
                      << "The sum is the last element of the decrypted vector" << std::endl;
            std::vector<int64_t> decrypted_sum_vec = sealhelper::decrypting(encrypted_sum_vec,
                                                                            analyst.he_sk,
                                                                            analyst_he_benc,
                                                                            *context,
                                                                            inputLen);
            // utils::print_vec(decrypted_sum_vec, decrypted_sum_vec.size(), "decrypted_sum_vec");
            int16_t decrypted_sum = decrypted_sum_vec[decrypted_sum_vec.size() - 1];
            std::cout << "The decrypted sum is " << decrypted_sum_vec[decrypted_sum_vec.size() - 1] << std::endl;
            if (vo_p1[i] != decrypted_sum)
            {
                throw std::runtime_error("Encrypted and plaintext results are different!");
            }
            i += 1;

            vo_he1.push_back(encrypted_sum_vec);
            // break;
        }
        std::cout << "TODO: CSP does the encrypted square activation on output of "
                  << "the encrypted fc1 layer on the HE data" << std::endl;

        std::cout << "TODO: CSP evaluates the encrypted fc2 layer output of "
                  << "the encrypted square activation layer" << std::endl;

        // std::cout << "(Check) Decrypt the vo_he1 to check" << std::endl;
        // std::vector<int64_t> vo_he_decrypted1;
        // for (const seal::Ciphertext &i : vo_he1)
        // {
        //     auto dec = sealhelper::decrypting(i, analyst.he_sk,
        //                                       analyst_he_benc, *context, inputLen);
        //     std::cout << dec[0] << " ";
        // }
        // utils::print_vec(vo_he_decrypted1, vo_he_decrypted1.size(), "vo_he_decrypted1");

        return 0;
    }

} // end of hhe_pktnn_examples namespace