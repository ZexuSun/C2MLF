import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class CustomDataset_outcome(Dataset):
    def __init__(self, data):
        super(CustomDataset_outcome, self).__init__()
        self.treatments = data['treatments']
        self.outcomes = data['outcomes']
        self.covariates = data['covariates']
        self.ivs = data['generate_ivs']
        self.hidden_confounders = data['generate_hidden_confounders']
        
    def __getitem__(self, index):
        treatments = self.treatments[index, 1:, :]
        outcome = self.outcomes[index, 1:, :]
        confounders = self.covariates[index, 1:, :]
        previous_treatments = self.treatments[index, :-1, :]
        hidden_confounders = self.hidden_confounders[index, 1:, :]
        confounders = np.concatenate([confounders, hidden_confounders], axis=-1)
        ivs = self.ivs[index, 1:, :]
        return previous_treatments, confounders, treatments, ivs, outcome
    
    def __len__(self):
        return len(self.treatments)  

class CustomDataset(Dataset):
    def __init__(self, data):
        super(CustomDataset, self).__init__()
        self.treatments = data['treatments']
        self.outcomes = data['outcomes']
        self.covariates = data['covariates']
    def __getitem__(self, index):
        treatments = self.treatments[index, :, :]
        previous_treatments = treatments[:-1, :]
        current_treatments = treatments[:, :]
        outcomes = self.outcomes[index, :, :]
        covariates = self.covariates[index, :, :]
        previous_covariates = covariates[:-1, :]
        current_covariates = covariates[:, :]
        return previous_covariates, previous_treatments, current_covariates, current_treatments, outcomes
    def __len__(self):
        return len(self.treatments)


def to_dict(df, treatment_features, confounder_features, outcome_features):
    data = dict()
    print('df.shape', df.shape)
    print('df[treatment_features].values.shape', df[treatment_features].values.shape)
    data['treatments'] = df[treatment_features].values.reshape(int(len(df)/14), 14, -1)
    data['outcomes'] = df[outcome_features].values.reshape(int(len(df)/14), 14, -1)
    data['covariates'] = df[confounder_features[0]].values.reshape(int(len(df)/14), 14, -1)
    for co_feature in confounder_features[1:]:
        data['covariates'] = np.concatenate((data['covariates'], df[co_feature].values.reshape(int(len(df)/14), 14, -1)), axis=-1)
    print('data[treatments].shape', data['treatments'].shape)
    print('data[outcomes].shape', data['outcomes'].shape)
    print('data[covariates].shape', data['covariates'].shape)
    return data

def factor_model_eval(data_dict, test_loader, factor_model):
    hidden_confounders, ivs = factor_model.compute_hidden_confounders_and_ivs(test_loader)
    data_dict['generate_hidden_confounders'] = hidden_confounders.reshape(-1, 14, 40)
    data_dict['generate_ivs'] = ivs.reshape(-1, 14, 40)
    return data_dict

def outcome_eval(outcome_test_loader, encoder, decoder):
    predicted_outcomes = []
    for val in range(-100, 2000, 1):
        val = val / 10
        treatment_value = [val, 1.7]
        # treatment_value = [val, 0.3]
        for i, (previous_treatments, current_confounders, current_treatments, current_ivs, outcomes) in enumerate(outcome_test_loader): 
            previous_treatments, current_confounders, current_treatments, outcomes = previous_treatments.float().cuda(1), current_confounders.float().cuda(1),\
                current_treatments.float().cuda(1), outcomes.float().cuda(1)
            print('current_confounders.shape', current_confounders.shape)
            print('previous_treatments.shape', previous_treatments.shape)
            _, init_states = encoder(current_confounders, previous_treatments)
            treatment_value = torch.tensor([treatment_value]).cuda(1)
            change_treatment = treatment_value.reshape(1, 1, 2)
            decoder_input = change_treatment
            init_states = init_states.unsqueeze(0)
            predicted_outcome = decoder(decoder_input, init_states)
            print('predicted_outcome', predicted_outcome.detach().cpu().numpy()[0,0,0])
            predicted_outcome = predicted_outcome.detach().cpu().numpy()[0,0,0]
            predicted_outcomes.append(predicted_outcome)

    return predicted_outcomes

def figure_plot(city_id, x_data, y_data):
    fig = plt.figure(figsize=(9,6))
    print('x_data', x_data)
    print('y_data', y_data)
    x_data_new = []
    for x in x_data:
        x_data_new.append(x* 0.0185 + 0.0139)
        #x_data_new.append(x* 0.0382 + 0.0288)
    y_data_new = []
    for y in y_data:
        y_data_new.append(y*0.444 + 0.0839)
    plt.plot(x_data_new, y_data_new, linestyle = '-')
    plt.title('City_id:{}'.format(city_id))
    plt.xlabel('pukuai_hufan_c_rate')
    plt.ylabel('tehui_single_delta_send_ratio')
    #plt.savefig('figs/result_5_260_{}.pdf'.format(city_id))
    plt.savefig('figs/result_pukuai_{}.png'.format(city_id))

def datadict_generation(total_df, city_id, start_date, end_date, treatment_features, confounder_features, outcome_features):
    df = total_df[(total_df['tdate'] >= start_date) & (total_df['tdate'] <= end_date)]
    df_city = df[df['city_id']==city_id]
    #sequence_num = len(df_city) // 14
    df_city = df_city[: 14]  
    print('df_city.shape', df_city.shape)
    test_data = to_dict(df_city, treatment_features, confounder_features, outcome_features)
    return test_data

def main():
    total_df = pd.read_csv('new_test_df.csv')
    city_id = 70
    print(total_df['city_id'])
    start_date = '2022-07-06'
    end_date = '2022-07-20'
    label = 'tehui_single_delta_send_ratio'
    confounder_features = ['is_holiday_yesterday', 'city_id', 'finish_cnt_14d', 'driver_cut_7d', 'month', 'online_rate_2d', 'driver_wait_time_rate_7d', 'finish_dri_num_3d', 'avg_bubble_cnt_21_d', 'avg_finish_cnt_7_d', 'driver_charge_time_h_2d', 'avg_gmv_per_charge_dis_7_d', 'online_listen_dri_num_21d', 'avg_driver_wait_dur_14d', 'grab_order_dur_21d', 'b_subsidy_3d', 'finish_cnt_21d', 'gmv_2d', 'strive_cnt_21d', 'avg_receive_dur_2d', 'avg_driver_wait_dur_7d', 'finish_dri_num_14d', 'avg_strive_cnt_14_d', 'online_listen_dri_num_3d', 'finish_cnt_7d', 'avg_call_pas_num_21_d', 'online_rate_3d', 'strive_cnt_7d', 'need_satisfy_rate_7d', 'avg_send_cnt_21_d', 'online_time_h_2d',  'week_of_year', 'online_rate_14d', 'gmv_3d', 'avg_receive_dur_3d', 'quarter', 'avg_online_rate_14_d', 'avg_avg_driver_wait_dur_7_d', 'online_time_h_21d', 'avg_online_rate_21_d', 'avg_grab_order_dur_21_d', 'avg_bubble_cnt_7_d', 'avg_online_time_h_7d', 'avg_bubble_cnt_14_d', 'avg_online_time_h_21_d', 'b_subsidy_21d', 'avg_receive_dis_14d',  'avg_online_listen_dri_num_21_d', 'need_satisfy_rate_3d', 'send_cnt_21d', 'avg_b_subsidy_14_d', 'avg_send_cnt_14_d', 'online_time_h_7d', 'avg_avg_charge_dis_14_d', 'driver_wait_time_rate_14d', 'avg_online_rate_7_d', 'avg_driver_cut_7_d', 'avg_finish_dri_num_14_d', 'send_cnt_3d', 'avg_finish_dri_num_21_d', 'avg_driver_cut_21_d', 'avg_avg_driver_wait_dur_21_d', 'gmv_21d', 'driver_cut_14d', 'need_satisfy_rate_14d', 'avg_avg_online_time_h_7_d', 'avg_online_time_h_21d', 'gmv_per_charge_dis_21d', 'driver_charge_time_h_21d', 'finish_dri_num_2d', 'avg_gmv_14_d', 'avg_charge_dis_3d', 'strive_time_rate_3d', 'call_pas_num_14d', 'online_time_h_14d', 'avg_finish_passenger_num_14_d', 'is_holiday', 'avg_b_subsidy_21_d', 'avg_online_time_h_14d', 'gmv_per_charge_dis_7d', 'strive_time_rate_21d', 'call_pas_num_7d', 'finish_passenger_num_7d', 'finish_dri_num_21d', 'avg_receive_dur_14d', 'avg_strive_cnt_21_d', 'avg_avg_driver_wait_dur_14_d', 'avg_call_pas_num_14_d', 'driver_wait_time_rate_21d', 'avg_avg_online_time_h_21_d', 'avg_online_listen_dri_num_7_d', 'avg_grab_order_dur_14_d', 'finish_passenger_num_3d', 'avg_gmv_21_d', 'avg_driver_cut_14_d', 'avg_driver_wait_dur_2d', 'driver_wait_time_rate_3d', 'avg_finish_dri_num_7_d', 'avg_driver_wait_dur_21d', 'b_subsidy_7d', 'avg_avg_receive_dur_14_d', 'strive_cnt_2d', 'finish_cnt_3d', 'need_satisfy_rate_21d', 'online_rate_7d', 'avg_charge_dis_2d', 'driver_cut_3d', 'avg_gmv_7_d', 'online_rate_21d', 'avg_call_pas_num_7_d', 'finish_dri_num_7d', 'need_satisfy_rate_2d', 'avg_charge_dis_7d', 'grab_order_dur_14d', 'avg_receive_dis_7d', 'avg_gmv_per_charge_dis_14_d', 'strive_time_rate_14d', 'avg_receive_dur_21d', 'avg_charge_dis_21d', 'finish_passenger_num_2d', 'online_listen_dri_num_7d', 'online_listen_dri_num_14d', 'strive_cnt_3d', 'avg_strive_cnt_7_d', 'driver_wait_time_rate_2d', 'avg_avg_charge_dis_21_d', 'online_listen_dri_num_2d', 'avg_avg_receive_dur_21_d', 'gmv_7d', 'finish_cnt_2d', 'avg_avg_online_time_h_14_d', 'day_of_week', 'driver_cut_2d', 'avg_finish_cnt_21_d', 'avg_receive_dur_7d', 'gmv_14d', 'send_cnt_2d', 'is_holiday_tomorrow', 'avg_gmv_per_charge_dis_21_d', 'avg_avg_charge_dis_7_d', 'finish_passenger_num_21d', 'avg_receive_dis_2d', 'avg_avg_receive_dur_7_d', 'avg_online_time_h_7_d', 'avg_charge_dis_14d', 'week_type', 'driver_charge_time_h_7d', 'avg_send_cnt_7_d', 'avg_b_subsidy_7_d','avg_online_time_h_3d', 'call_pas_num_21d', 'avg_online_listen_dri_num_14_d', 'avg_receive_dis_3d', 'send_cnt_14d', 'gmv_per_charge_dis_2d', 'avg_avg_receive_dis_21_d', 'online_time_h_3d', 'grab_order_dur_3d', 'avg_finish_passenger_num_21_d', 'avg_finish_cnt_14_d', 'strive_time_rate_2d', 'avg_online_time_h_14_d', 'driver_charge_time_h_3d', 'b_subsidy_2d', 'driver_cut_21d', 'strive_time_rate_7d', 'b_subsidy_14d', 'avg_avg_receive_dis_7_d', 'call_pas_num_2d', 'avg_finish_passenger_num_7_d', 'driver_charge_time_h_14d',  'avg_online_time_h_2d', 'gmv_per_charge_dis_14d', 'avg_driver_wait_dur_3d', 'avg_grab_order_dur_7_d', 'gmv_per_charge_dis_3d', 'grab_order_dur_7d', 'avg_avg_receive_dis_14_d', 'grab_order_dur_2d', 'send_cnt_7d', 'call_pas_num_3d', 'avg_receive_dis_21d', 'strive_cnt_14d', 'finish_passenger_num_14d', 'avg_bubble_cnt_2_d', 'avg_bubble_cnt_3_d', 'avg_send_cnt_2_d', 'avg_send_cnt_3_d', 'avg_strive_cnt_2_d', 'avg_strive_cnt_3_d', 'avg_finish_cnt_2_d', 'avg_finish_cnt_3_d', 'avg_call_pas_num_2_d', 'avg_call_pas_num_3_d', 'avg_finish_passenger_num_2_d', 'avg_finish_passenger_num_3_d', 'avg_grab_order_dur_2_d', 'avg_grab_order_dur_3_d', 'avg_avg_receive_dur_2_d', 'avg_avg_receive_dur_3_d', 'avg_avg_receive_dis_2_d', 'avg_avg_receive_dis_3_d', 'avg_avg_driver_wait_dur_2_d', 'avg_avg_driver_wait_dur_3_d', 'avg_gmv_2_d', 'avg_gmv_po_2_d', 'avg_gmv_per_charge_dis_2_d', 'avg_gmv_3_d', 'avg_gmv_po_3_d', 'avg_gmv_per_charge_dis_3_d', 'avg_gmv_po_7_d', 'avg_gmv_po_14_d', 'avg_gmv_po_21_d', 'avg_c_subsidy_po_2_d', 'avg_c_subsidy_rate_2_d', 'avg_c_subsidy_po_3_d', 'avg_c_subsidy_rate_3_d', 'avg_c_subsidy_po_7_d', 'avg_c_subsidy_rate_7_d', 'avg_c_subsidy_po_14_d', 'avg_c_subsidy_rate_14_d', 'avg_c_subsidy_po_21_d', 'avg_c_subsidy_rate_21_d', 'avg_b_subsidy_2_d', 'avg_b_subsidy_po_2_d', 'avg_b_subsidy_rate_2_d', 'avg_b_subsidy_3_d', 'avg_b_subsidy_po_3_d', 'avg_b_subsidy_rate_3_d', 'avg_b_subsidy_po_7_d', 'avg_b_subsidy_rate_7_d', 'avg_b_subsidy_po_14_d', 'avg_b_subsidy_rate_14_d', 'avg_b_subsidy_po_21_d', 'avg_b_subsidy_rate_21_d', 'avg_online_listen_dri_num_2_d', 'avg_online_listen_dri_num_3_d', 'avg_online_rate_2_d', 'avg_online_rate_3_d', 'avg_online_time_h_2_d', 'avg_avg_online_time_h_2_d', 'avg_online_time_h_3_d', 'avg_avg_online_time_h_3_d', 'avg_finish_dri_num_2_d', 'avg_finish_dri_num_3_d', 'avg_driver_cut_2_d', 'avg_driver_cut_3_d', 'avg_avg_charge_dis_2_d', 'avg_avg_charge_dis_3_d', 'avg_avg_finish_order_cnt_2_d', 'avg_avg_finish_order_cnt_3_d', 'avg_avg_finish_order_cnt_7_d', 'avg_avg_finish_order_cnt_14_d', 'avg_avg_finish_order_cnt_21_d', 'avg_finish_order_cnt_2d', 'avg_finish_order_cnt_3d', 'avg_finish_order_cnt_7d', 'avg_finish_order_cnt_14d', 'avg_finish_order_cnt_21d', 'avg_need_satisfy_rate_2_d', 'avg_need_satisfy_rate_3_d', 'avg_need_satisfy_rate_7_d', 'avg_need_satisfy_rate_14_d', 'avg_need_satisfy_rate_21_d', 'gmv_po_2d', 'gmv_po_3d', 'gmv_po_7d', 'gmv_po_14d', 'gmv_po_21d',
              'avg_%s_2_d'%label, 'avg_%s_3_d'%label, 'avg_%s_7_d'%label, 'avg_%s_14_d'%label, 'avg_%s_21_d'%label, 'avg_%s_28_d'%label, 'avg_%s_30_d'%label,
              '%s_2d'%label, '%s_3d'%label, '%s_7d'%label, '%s_14d'%label, '%s_21d'%label, '%s_28d'%label, '%s_30d'%label,
              'c_subsidy_po_2d', 'c_subsidy_rate_2d', 'c_subsidy_po_3d', 'c_subsidy_rate_3d', 'c_subsidy_po_7d', 'c_subsidy_rate_7d', 'c_subsidy_po_14d', 'c_subsidy_rate_14d', 'c_subsidy_po_21d', 'c_subsidy_rate_21d', 'b_subsidy_po_2d', 'b_subsidy_rate_2d', 'b_subsidy_po_3d', 'b_subsidy_rate_3d', 'b_subsidy_po_7d', 'b_subsidy_rate_7d', 'b_subsidy_po_14d', 'b_subsidy_rate_14d', 'b_subsidy_po_21d', 'b_subsidy_rate_21d', 'avg_strive_time_rate_2_d', 'avg_strive_time_rate_3_d', 'avg_strive_time_rate_7_d', 'avg_strive_time_rate_14_d', 'avg_strive_time_rate_21_d', 'avg_driver_wait_time_rate_2_d', 'avg_driver_wait_time_rate_3_d', 'avg_driver_wait_time_rate_7_d', 'avg_driver_wait_time_rate_14_d', 'avg_driver_wait_time_rate_21_d', 'avg_driver_charge_time_h_2_d', 'avg_driver_charge_time_h_3_d', 'avg_driver_charge_time_h_7_d', 'avg_driver_charge_time_h_14_d', 'avg_driver_charge_time_h_21_d',
               'history_city_%s_min'%label, 'history_city_%s_max'%label, 'history_city_%s_mean'%label, 'history_city_%s_var'%label,
              'cluster_c_combine', 'history_cluster_%s_min' % label, 'history_cluster_%s_max' % label, 'history_cluster_%s_mean' % label, 'history_cluster_%s_var' % label,
             'cy_daily_cloudrate_avg_0', 'cy_daily_wind_avg_speed_0','cy_daily_temperature_avg_0', 'cy_daily_humidity_avg_0', 'cy_daily_pres_avg_0', 'cy_daily_aqi_avg_0','cy_daily_temperature_max_0', 'cy_daily_precipitation_avg_0', 'cy_daily_temperature_min_0', 'cy_daily_visibility_avg_0','cy_daily_comfort_index_0', 'cy_daily_pm25_avg_0', 'cy_daily_wind_avg_direction_0'
              ] + ['gongxu','gmv_po'] #泛快非拼真实供需
    print('len(confounder_features)', len(confounder_features))
    outcome_features = ['tehui_single_delta_send_ratio']
    treatment_features = ['tehui_hufan_c_rate', 'pukuai_hufan_c_rate']
    print('total_df.shape', total_df.shape)
    #print('min', total_df['subsidy_c_rate'].min())
    #print('max', total_df['subsidy_c_rate'].max())

    factor_model = torch.load('factor_model.pt', map_location='cuda:1')
    encoder = torch.load('outcome_encoder.pt', map_location='cuda:1')
    decoder = torch.load('outcome_decoder.pt', map_location='cuda:1')
    city_data = datadict_generation(total_df, city_id, start_date, end_date, treatment_features, confounder_features, outcome_features)
    factor_custom = CustomDataset(city_data)
    factor_loader = DataLoader(factor_custom, batch_size=1, shuffle=False)
    new_dict = factor_model_eval(city_data, factor_loader, factor_model)
    outcome_custom = CustomDataset_outcome(new_dict)
    outcome_loader = DataLoader(outcome_custom, batch_size=1, shuffle=False)
    predicted_outcomes = outcome_eval(outcome_loader, encoder, decoder)
    x_data = np.arange(-10, 200, 0.1)
    y_data = predicted_outcomes
    figure_plot(city_id, x_data, y_data)   

if __name__ == '__main__':
    main()



