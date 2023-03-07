import pickle
import json
import numpy as np
import os 

class Prediction():
    def __init__(self):
        print(os.getcwd())
        

    def load_raw(self):
        with open(r'D:\V\01 Oct\23_03_06_ml_project_flask\new_project\app\linear_model.pkl','rb') as model_file: 
            self.model = pickle.load(model_file)
        
        with open(r'D:\V\01 Oct\23_03_06_ml_project_flask\new_project\app\columns_names.json','r') as col_file: 
            self.column_names = json.load(col_file)
            
        with open(r'D:\V\01 Oct\23_03_06_ml_project_flask\new_project\app\encoded_data.json','r') as encode_file:
            self.encoded_data = json.load(encode_file)

        print(f"we are in load raw")

    def predict_price(self,data):
       
        self.load_raw()
        self.data = data
        user_input = np.zeros(len(self.column_names['Column Names']))
        array = np.array(self.column_names['Column Names'])
        symboling = self.data['html_symb']
        normalized_losses = self.data['html_norm']
        make = self.data['html_make']
        fuel_type = self.data['html_fuel']
        aspiration = self.data['html_asp']
        num_of_doors = self.data['html_no_of_doors']
        body_style = self.data['html_body']
        drive_wheels = self.data['html_drive']
        engine_location = self.data['html_doption']
        wheel_base = self.data['html_wb']
        length = self.data['html_length']
        width = self.data['html_width']
        height = self.data['html_height']
        curb_weight = self.data['html_cw'] 
        engine_type = 'ohc'
        num_of_cylinders = 'four'
        engine_size =  141
        fuel_system = 'mpfi'
        bore = 3.78
        stroke = 3.15
        compression_ratio = 9.5 
        horsepower =  115
        peak_rpm = 5900
        city_mpg = 19
        highway_mpg = 24 


        user_input[0] = symboling
        user_input[1] = normalized_losses

        make_string = 'make_'+make
        make_index = np.where(array == make_string)[0][0]
        user_input[make_index] = 1 

        user_input[2] = int(fuel_type)
        user_input[3] = int(aspiration)
        user_input[4] = int(num_of_doors)

        body_style_string = 'body-style_'+body_style
        bs_index = np.where(array == body_style_string)[0][0]
        user_input[bs_index] = 1

        drive_wheels_string = "drive-wheels_"+drive_wheels
        ds_index = np.where(array ==drive_wheels_string)[0][0]
        user_input[ds_index] = 1

        user_input[5] = int(engine_location)
        user_input[6] = wheel_base
        user_input[7] = length 
        user_input[8] = width 
        user_input[9] = height 
        user_input[10] = curb_weight

        engine_type_string = "engine-type_"+engine_type
        et_index = np.where(array ==engine_type_string )[0][0]
        user_input[et_index] = 1

        user_input[11] = self.encoded_data['num_of_cylinders'][num_of_cylinders]
        user_input[12] = engine_size

        fuel_system_string = "fuel-system_"+fuel_system
        fs_index = np.where(array == fuel_system_string)[0][0]
        user_input[fs_index] = 1

        user_input[13] = bore
        user_input[14] = stroke 
        user_input[15] = compression_ratio 
        user_input[16] = horsepower 
        user_input[17] = peak_rpm 
        user_input[18] = city_mpg 
        user_input[19] = highway_mpg 

        print(f"{user_input=}")
        print(len(user_input))

        price = self.model.predict([user_input])
        print(f"Predicted Price = {price}")

        return price
    
if __name__ == "__main__":
 
    pred_obj = Prediction()
    pred_obj.load_raw()