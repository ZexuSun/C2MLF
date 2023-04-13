from scipy.special import expit

import numpy as np
import joblib

class AutoregressiveSimulation:
    def __init__(self, hidden_confounding_degree, iv_degree, num_simulated_hidden_confounders, num_simulated_ivs, type):
        self.num_covariates = 20
        self.num_confounders = num_simulated_hidden_confounders
        self.num_treatments = 1
        self.num_ivs = num_simulated_ivs
        self.p = 5

        self.type = type
        self.phi_x = 0.5
        self.gamma_x = 0.5
        self.gamma_a = 0.5
        self.phi_l = hidden_confounding_degree
        self.gamma_l = hidden_confounding_degree
        self.phi_z = iv_degree

        self.covariates_coefficients = dict()
        self.covariates_coefficients['treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_covariates, self.num_treatments), treatment_coefficients=True)

        self.covariates_coefficients['covariates'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_covariates, self.num_covariates), variables_coefficients=True)

        self.confounders_coefficients = dict()
        self.confounders_coefficients['treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_treatments))
        self.confounders_coefficients['confounders'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_confounders), variables_coefficients=True)
        
        self.ivs_coefficients = dict()
        self.ivs_coefficients['treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_ivs, self.num_treatments))
        self.ivs_coefficients['ivs'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_ivs, self.num_ivs), variables_coefficients=True)

        self.outcome_coefficients = np.array([np.random.normal(0, 1) for _ in range(self.num_confounders + self.num_covariates + self.num_treatments)])
        self.treatment_coefficients = self.generate_treatment_coefficients()


    def generate_treatment_coefficients(self):
        treatment_coefficients = np.zeros(shape=(self.num_treatments, self.num_covariates + self.num_confounders + self.num_ivs))
        for treatment in range(self.num_treatments):
            treatment_coefficients[treatment][self.num_ivs] = self.phi_z
            treatment_coefficients[treatment][self.num_covariates] = self.phi_x
            treatment_coefficients[treatment][self.num_confounders] = self.phi_l

        return treatment_coefficients


    def generate_coefficients(self, p, matrix_shape, variables_coefficients=False, treatment_coefficients=False):
        coefficients = []
        for i in range(p):
            if (variables_coefficients):
                diag_elements = [np.random.normal(1.0 - (i+1) * 0.2, 0.2) for _ in range(matrix_shape[0])]
                timestep_coefficients = np.diag(diag_elements)

            elif (treatment_coefficients):
                diag_elements = [np.random.normal(0, 0.5)  for _ in range(matrix_shape[1])]
                timestep_coefficients = np.diag(diag_elements)
            else:
                timestep_coefficients = np.random.normal(0, 0.5, size=matrix_shape[1])

            normalized_coefficients = timestep_coefficients / p
            coefficients.append(normalized_coefficients)

        return coefficients

    def generate_treatment_assignments_single_timestep(self, p, history):
        confounders_history = history['confounders']
        covariates_history = history['covariates']
        ivs_history = history['ivs']
        history_length = len(covariates_history)
        if (history_length < p):
            p = history_length

        average_covariates = np.zeros(shape=len(covariates_history[-1]))
        average_confounders = np.zeros(shape=len(confounders_history[-1]))
        average_ivs = np.zeros(shape=len(confounders_history[-1]))
        for index in range(p):
            average_covariates = average_covariates + covariates_history[history_length - index - 1]
            average_confounders = average_confounders + confounders_history[history_length - index - 1]
            average_ivs = average_ivs + ivs_history[history_length - index - 1]

        all_variables = np.concatenate((average_covariates, average_confounders, average_ivs)).T

        treatment_assignment = np.zeros(shape=(self.num_treatments,))
        for index in range(self.num_treatments):
            aux_normal = np.dot(all_variables, self.treatment_coefficients[index])
            if self.type == 'train':
                #treatment_assignment[index] = np.abs(np.random.normal(1 * expit(aux_normal), 0.1))
                treatment_assignment[index] = np.random.normal(1 * expit(aux_normal), 0.1)
                treatment_assignment[index] = np.clip(treatment_assignment[index], 0, 2)
            else: 
                treatment_assignment[index] = np.random.uniform(0, 2)
        return treatment_assignment

    def generate_covariates_single_timestep(self, p, history):
        treatments_history = history['treatments']
        covariates_history = history['covariates']

        past_treatment_coefficients = self.covariates_coefficients['treatments']
        past_covariates_coefficients = self.covariates_coefficients['covariates']

        history_length = len(covariates_history)
        if (history_length < p):
            p = history_length

        treatments_sum = np.zeros(shape=(self.num_covariates,))
        covariates_sum = np.zeros(shape=(self.num_covariates,))
        for index in range(p):
            treatments_sum += np.matmul(past_treatment_coefficients[index],
                                        treatments_history[history_length - index - 1])

            covariates_sum += np.matmul(past_covariates_coefficients[index],
                                        covariates_history[history_length - index - 1])

        noise = np.random.normal(0, 0.01, size=(self.num_covariates))

        x_t = treatments_sum + covariates_sum + noise
        x_t = np.clip(x_t, -1, 1)

        return x_t

    def generate_confounders_single_timestep(self, p, history):
        treatments_history = history['treatments']
        confounders_history = history['confounders']

        past_treatment_coefficients = self.confounders_coefficients['treatments']
        past_confounders_coefficients = self.confounders_coefficients['confounders']

        history_length = len(confounders_history)
        if (history_length < p):
            p = history_length

        treatments_sum = np.zeros(shape=(self.num_confounders,))
        confounders_sum = np.zeros(shape=(self.num_confounders,))
        for index in range(p):
            treatments_sum += np.matmul(past_treatment_coefficients[index],
                                        treatments_history[history_length - index - 1])
            confounders_sum += np.matmul(past_confounders_coefficients[index],
                                         confounders_history[history_length - index - 1])

        noise = np.random.normal(0, 0.01, size=(self.num_confounders))

        l_t = treatments_sum + confounders_sum + noise
        l_t = np.clip(l_t, -1, 1)

        return l_t

    def generate_ivs_single_timestep(self, p, history):
        treatments_history = history['treatments']
        ivs_history = history['ivs']

        past_treatment_coefficients = self.confounders_coefficients['treatments']
        past_ivs_coefficients = self.ivs_coefficients['ivs']

        history_length = len(ivs_history)
        if (history_length < p):
            p = history_length

        treatments_sum = np.zeros(shape=(self.num_ivs,))
        ivs_sum = np.zeros(shape=(self.num_ivs,))
        for index in range(p):
            treatments_sum += np.matmul(past_treatment_coefficients[index],
                                        treatments_history[history_length - index - 1])
            ivs_sum += np.matmul(past_ivs_coefficients[index],
                                         ivs_history[history_length - index - 1])

        noise = np.random.normal(0, 0.01, size=(self.num_ivs))

        z_t = treatments_sum + ivs_sum + noise
        z_t = np.clip(z_t, -1, 1)

        return z_t

    def generate_data_single_patient(self, timesteps):

        x_0 = np.random.normal(0, 2, size=(self.num_covariates,))
        l_0 = np.random.normal(0, 2, size=(self.num_confounders,))
        z_0 = np.random.normal(0, 2, size=(self.num_ivs,))
        a_0 = np.zeros(shape=(self.num_treatments,))


        history = dict()
        history['covariates'] = [x_0]
        history['confounders'] = [l_0]
        history['ivs'] = [z_0]
        history['treatments'] = [a_0]

        for t in range(timesteps):
            x_t = self.generate_covariates_single_timestep(self.p, history)
            l_t = self.generate_confounders_single_timestep(self.p, history)
            z_t = self.generate_ivs_single_timestep(self.p, history)
            history['covariates'].append(x_t)
            history['confounders'].append(l_t)
            history['ivs'].append(z_t)

            a_t = self.generate_treatment_assignments_single_timestep(self.p, history)
            
            history['treatments'].append(a_t)

        return np.array(history['covariates']), np.array(history['confounders']), np.array(history['ivs']), np.array(history['treatments'])

    def generate_dict_dataset(self, num_patients, timesteps, p):
        dataset = dict()
        for patient in range(num_patients):
            covariates_history, confounders_history, ivs_history, treatments_history = self.generate_data_single_patient(timesteps,
                                                                                                            p)
            dataset[patient] = dict()
            dataset[patient]['previous_covariates'] = np.array(covariates_history[0:timesteps - 1])
            dataset[patient]['previous_treatments'] = np.array(treatments_history[0:timesteps - 1])

            dataset[patient]['previous_ivs'] = np.array(ivs_history[0:timesteps - 1])

            dataset[patient]['covariates'] = np.array(covariates_history[1:timesteps])
            dataset[patient]['confounders'] = np.array(confounders_history[1:timesteps])
            dataset[patient]['ivs'] = np.array(ivs_history[1:timesteps])
            dataset[patient]['treatments'] = np.array(treatments_history[1:timesteps])

        return dataset


    # def normalize_dataset(self, dataset):

    #     for covariate_id in range(self.num_covariates):
    #         covariate_mean = np.mean(dataset['previous_covariates'][:, :, covariate_id])
    #         covariate_std = np.std(dataset['previous_covariates'][:, :, covariate_id])
    #         dataset['previous_covariates'][:, :, covariate_id] = (dataset['previous_covariates'][:, :,
    #                                                   covariate_id] - covariate_mean) / covariate_std

    #     for covariate_id in range(self.num_covariates):
    #         covariate_mean = np.mean(dataset['covariates'][:, :, covariate_id])
    #         covariate_std = np.std(dataset['covariates'][:, :, covariate_id])
    #         dataset['covariates'][:, :, covariate_id] = (dataset['covariates'][:, :,
    #                                                   covariate_id] - covariate_mean) / covariate_std

    #     for confounder_id in range(self.num_confounders):
    #         confounder_mean = np.mean(dataset['confounders'][:, :, confounder_id])
    #         confounder_std = np.std(dataset['confounders'][:, :, confounder_id])
    #         dataset['confounders'][:, :, confounder_id] = (dataset['confounders'][:, :,
    #                                                      confounder_id] - confounder_mean) / confounder_std

    #     outcome_mean = np.mean(dataset['outcomes'])
    #     outcome_std= np.std(dataset['outcomes'])
    #     dataset['outcomes'] = (dataset['outcomes'] - outcome_mean) /outcome_std

    #     return dataset


    def generate_dataset(self, num_patients, max_timesteps, binary_outcome=False):
            dataset = dict()

            dataset['previous_covariates'] = []
            dataset['previous_treatments'] = []
            dataset['covariates'] = []
            dataset['confounders'] = []
            dataset['treatments'] = []
            dataset['sequence_length'] = []
            dataset['outcomes'] = []
            dataset['ivs'] = []

            for patient in range(num_patients):
                timesteps = 17#np.random.randint(int(max_timesteps)-10, int(max_timesteps), 1)[0]
                covariates_history, confounders_history, ivs_history, treatments_history = self.generate_data_single_patient(
                    timesteps + 1)

                previous_covariates = np.vstack((np.array(covariates_history[1:timesteps - 1]),
                                                 np.zeros(shape=(max_timesteps-timesteps, self.num_covariates))))
                previous_treatments = np.vstack((np.array(np.array(treatments_history[1:timesteps - 1])),
                                                 np.zeros(shape=(max_timesteps-timesteps, self.num_treatments))))

                covariates = np.vstack((np.array(covariates_history[1:timesteps]),
                                        np.zeros(shape=(max_timesteps - timesteps, self.num_covariates))))

                confounders = np.vstack((np.array(confounders_history[1:timesteps]),
                                        np.zeros(shape=(max_timesteps - timesteps, self.num_confounders))))

                #treatments = np.abs(np.vstack((np.array(treatments_history[1:timesteps]),
                #                        np.zeros(shape=(max_timesteps-timesteps, self.num_treatments)))))
                treatments = np.vstack((np.array(treatments_history[1:timesteps]),
                                        np.zeros(shape=(max_timesteps-timesteps, self.num_treatments))))

                ivs = np.vstack((np.array(ivs_history[1:timesteps]),
                                        np.zeros(shape=(max_timesteps-timesteps, self.num_ivs))))


                outcomes = self.gamma_l * np.mean(np.array(confounders_history[2:timesteps + 1]), axis=-1) + \
                           self.gamma_x * np.mean(np.array(covariates_history[2:timesteps + 1]), axis=-1) + \
                           self.gamma_a * np.mean(np.array(treatments_history[2:timesteps + 1]), axis=-1)

                outcomes = outcomes[:, np.newaxis]
                outcomes = np.vstack((np.array(outcomes),
                                        np.zeros(shape=(max_timesteps-timesteps, 1))))

                dataset['previous_covariates'].append(np.array(previous_covariates))
                dataset['previous_treatments'].append(np.array(previous_treatments))
                dataset['covariates'].append(np.array(covariates))
                dataset['confounders'].append(np.array(confounders))
                dataset['treatments'].append(np.array(treatments))
                dataset['sequence_length'].append(np.array(timesteps))
                dataset['ivs'].append(np.array(ivs))
                dataset['outcomes'].append(np.array(outcomes))


            for key in dataset.keys():
                dataset[key] = np.array(dataset[key])

            return dataset


autoregressive_train = AutoregressiveSimulation(0.5, 0.5, 10, 10, 'train')
train_dataset = autoregressive_train.generate_dataset(8000, 17)

autoregressive_val = AutoregressiveSimulation(0.5, 0.5, 10, 10, 'train')
val_dataset = autoregressive_val.generate_dataset(1000, 17)

autoregressive_test = AutoregressiveSimulation(0.5, 0.5, 10, 10, 'test')
test_dataset = autoregressive_test.generate_dataset(1000, 17)
print('finished')

joblib.dump(train_dataset, 'simulate_0.5_0.5/train_simulate_data_16.pkl')
joblib.dump(val_dataset, 'simulate_0.5_0.5/val_simulate_data_16.pkl')
joblib.dump(test_dataset, 'simulate_0.5_0.5/test_simulate_data_16.pkl')
#print(train_dataset['treatments'])
print(train_dataset['treatments'].shape)
print(val_dataset['treatments'].shape)
print(test_dataset['treatments'].shape)