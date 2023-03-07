import numpy as np
import pickle
import json


with open('linear_model.pkl','rb') as model_file: 
    model = pickle.load(model_file)
    
with open('columns_names.json','r') as col_file: 
    column_names = json.load(col_file)
    
with open('encoded_data.json','r') as encode_file:
    encoded_data = json.load(encode_file)

user_input = np.zeros(len(column_names['Column Names']))

array = np.array(column_names['Column Names'])

symboling = -2  # API
normalized_losses = 45
make = 'isuzu'
fuel_type = 'gas'
aspiration = 'turbo'
num_of_doors = 'two'
body_style = 'sedan'
drive_wheels = 'rwd'
engine_location = 'front'
wheel_base = 90
length = 180
width = 60
height = 50
curb_weight = 2000 
engine_type = 'ohc'
num_of_cylinders = 'four'
engine_size =  120
fuel_system = 'mpfi'
bore = 3.78
stroke = 3.15
compression_ratio = 9.5 
horsepower =  115
peak_rpm = 3400
city_mpg = 19
highway_mpg = 24 

user_input[0] = symboling
user_input[1] = normalized_losses

make_string = 'make_'+make
make_index = np.where(array == make_string)[0][0]
user_input[make_index] = 1 

user_input[2] = encoded_data['fuel_type'][fuel_type]
user_input[3] = encoded_data['aspiration'][aspiration]
user_input[4] = encoded_data['num_of_doors'][num_of_doors]

body_style_string = 'body-style_'+body_style
bs_index = np.where(array == body_style_string)[0][0]
user_input[bs_index] = 1

drive_wheels_string = "drive-wheels_"+drive_wheels
ds_index = np.where(array ==drive_wheels_string)[0][0]
user_input[ds_index] = 1

user_input[5] = encoded_data['engine_location'][engine_location]
user_input[6] = wheel_base
user_input[7] = length 
user_input[8] = width 
user_input[9] = height 
user_input[10] = curb_weight

engine_type_string = "engine-type_"+engine_type
et_index = np.where(array ==engine_type_string )[0][0]
user_input[et_index] = 1

user_input[11] = encoded_data['num_of_cylinders'][num_of_cylinders]
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

price = model.predict([user_input])
print(f"Predicted Price = {price}")
print(f"Actual Price = 22625")