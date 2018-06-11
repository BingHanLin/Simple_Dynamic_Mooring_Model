import json
import math
import numpy as np
import pandas as pd
class OCEAN:
    def __init__(self, filename):

        with open(filename,'r') as load_f:
            params_dict = json.load(load_f)

        self.wave_period        = params_dict["OCEAN"]["WavePeriod"]
        self.wave_height        = params_dict["OCEAN"]["WaveHeight"]
        self.water_depth        = params_dict["OCEAN"]["WaterDepth"]
        self.water_density      = params_dict["OCEAN"]["WaterDensity"]
        self.water_viscosity    = params_dict["OCEAN"]["WaterViscosity"]

        self.wave_angle_in      = params_dict["OCEAN"]["InWaveAngle"]
        self.current_angle_in   = params_dict["OCEAN"]["InCurrentAngle"]      
        self.current_velocity   = params_dict["OCEAN"]["InCurrentVel"]

        self.gravity            = params_dict["OCEAN"]["Gravity"]
        
        self.__cal_const_var()
        
        print ("Ocean condition is built.")

    # =======================================
    # 計算波浪相關常數
    # =======================================
    def __cal_const_var(self):
        
        # wave frequency
        self.__sigma = 2*math.pi/self.wave_period

        # obtain wave number, apparent angular frequency 
        # by newton's method
        wk0 = 2*math.pi/(1.56*self.wave_period**2) 
        sigmae0 = self.__sigma-self.current_velocity*wk0
        wk1 = sigmae0**2/(self.gravity*math.tanh(wk0*self.water_depth))

        while ( abs(wk1-wk0) > 10e-5):
            wk0 = wk1
            wk1 = sigmae0**2 / (self.gravity*math.tanh(wk0*self.water_depth))
            sigmae0 = self.__sigma - self.current_velocity*wk0
        
        # apparent angular frequency
        self.__sigmae = sigmae0

        # wave number
        self.__wk = wk0

        # wave velocity
        self.__wc = self.__sigma/self.__wk 

        # wave number in x, y direction
        self.__kx=self.__wk*math.cos(self.wave_angle_in)
        self.__ky=self.__wk*math.sin(self.wave_angle_in)

        # current velocity in x, y direction
        self.current_velocity_x = self.current_velocity*math.cos(self.current_angle_in)
        self.current_velocity_y = self.current_velocity*math.sin(self.current_angle_in)

    # =======================================
    # 計算波流場
    # =======================================
    def cal_wave_field(self, node_position, time, reduction = 1):


        phase = self.__kx*node_position[0,:] + self.__ky*node_position[1,:] - self.__sigma*time
        
        self.eta = np.where(self.__kx*node_position[0,:] < self.__sigma*time, 0.5*self.wave_height*np.sin(phase) , 0) 

        num_node = len(node_position[0,:])
        water_velocity = np.zeros((3, num_node))
        water_acc = np.zeros((3, num_node))


        condition = ( (self.__kx*node_position[0,:] < self.__sigma*time) * (self.eta > node_position[2,:]) )


        Amp = 0.5*self.wave_height*self.gravity/(self.__sigmae*math.cosh(self.__wk*self.water_depth))

        # water_velocity[0,:] = np.where( condition , self.current_velocity_x*reduction+Amp*self.__kx*np.cosh(self.__wk*(self.water_depth+node_position[2,:]))*np.sin(phase) , 0)   
        # water_velocity[1,:] = np.where( condition , self.current_velocity_y*reduction+Amp*self.__ky*np.cosh(self.__wk*(self.water_depth+node_position[2,:]))*np.sin(phase) , 0)   
        # water_velocity[2,:] = np.where( condition , -Amp*self.__wk*np.sinh(self.__wk*(self.water_depth+node_position[2,:]))*np.cos(phase), 0) 

        # water_acc[0,:] = np.where( condition , -self.__sigma*Amp*self.__kx*np.cosh(self.__wk*(self.water_depth+node_position[2,:]))*np.cos(phase)    , 0)   
        # water_acc[1,:] = np.where( condition , -self.__sigma*Amp*self.__ky*np.cosh(self.__wk*(self.water_depth+node_position[2,:]))*np.cos(phase) , 0)   
        # water_acc[2,:] = np.where( condition , -self.__sigma*Amp*self.__wk*np.sinh(self.__wk*(self.water_depth+node_position[2,:]))*np.sin(phase), 0) 

        water_velocity[0,:] = self.current_velocity_x*reduction 
        water_velocity[1,:] = self.current_velocity_y*reduction
        water_velocity[2,:] = self.current_velocity_x*reduction

        water_acc[0,:] = -self.__sigma*Amp*self.__kx*np.cosh(self.__wk*(self.water_depth+node_position[2,:]))*np.cos(phase)   
        water_acc[1,:] = -self.__sigma*Amp*self.__ky*np.cosh(self.__wk*(self.water_depth+node_position[2,:]))*np.cos(phase)  
        water_acc[2,:] = -self.__sigma*Amp*self.__wk*np.sinh(self.__wk*(self.water_depth+node_position[2,:]))*np.sin(phase)

        return water_velocity, water_acc


    def plot_ocean(self, x_range, y_range, x_num, y_num, time, ax):

        nx, ny = (x_num, y_num)
        x = np.linspace( x_range[0], x_range[1], nx)
        y = np.linspace( y_range[0], y_range[1], ny)
        xv, yv = np.meshgrid(x, y)


        eta = np.zeros((ny,nx))
        water_depth = np.zeros((ny,nx))
        for i in range(ny):
            for j in range(nx):
                self.cal_wave_field(xv[i,j], yv[i,j], 0, time, 0)
                eta[i,j] = self.eta
                water_depth[i,j] = -self.water_depth

        ax.plot_surface(xv,yv, eta, linewidth=0, alpha=0.8,)
        ax.plot_surface(xv,yv, water_depth, linewidth=0)


    def save_data_csv(self, present_time, DirName):
        
        FileName = './'+ DirName + '/' + 'ocean_data'+ str("%.5f" % (present_time))+'.csv'
        
        xv, yv = np.mgrid[-10:130:5, -10:30:5]
        zv = np.zeros_like(xv)
        pos = np.stack((xv.ravel(), yv.ravel(), zv.ravel()) )

        self.cal_wave_field(pos, present_time, 0)

        datadict = {}
        datadict['x'] = xv.ravel()
        datadict['y'] = yv.ravel()
        datadict['eta'] = self.eta

        # make dataframe
        dataframe = pd.DataFrame(datadict)

        dataframe.to_csv(FileName, sep=',', mode='a')



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    oceana = OCEAN("Params.json")


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    water_velocity, water_acc = oceana.cal_wave_field(np.asarray([[10],[2],[-10]]),100)
    print (water_velocity)
    # plt.show()