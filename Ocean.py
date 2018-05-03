import json
import math
import numpy as np

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
    def cal_wave_field(self, xp, yp, zp, time, reduction = 1):
        
        phase = self.__kx*xp + self.__ky*yp - self.__sigma*time

        self.eta = 0.5*self.wave_height*math.sin(phase)

        # 用這個? 
        # if ( (self.__kx*xp < self.__sigma*time) and (self.eta > zp) ):
        if ( (self.eta > zp) ):

            A = 0.5*self.wave_height*self.gravity/(self.__sigmae*math.cosh(self.__wk*self.water_depth))

            wu = self.current_velocity_x*reduction+A*self.__kx*math.cosh(self.__wk*(self.water_depth+zp))*math.sin(phase)
            wv = self.current_velocity_y*reduction+A*self.__ky*math.cosh(self.__wk*(self.water_depth+zp))*math.sin(phase)
            ww = -A*self.__wk*math.sinh(self.__wk*(self.water_depth+zp))*math.cos(phase)	   
            
            wax = -self.__sigma*A*self.__kx*math.cosh(self.__wk*(self.water_depth+zp))*math.cos(phase)   
            way = -self.__sigma*A*self.__ky*math.cosh(self.__wk*(self.water_depth+zp))*math.cos(phase)
            waz = -self.__sigma*A*self.__wk*math.sinh(self.__wk*(self.water_depth+zp))*math.sin(phase)	

        else:

            wu=0
            wv=0
            ww=0

            wax=0
            way=0
            waz=0

        water_velocity = np.asarray([wu, wv, ww])
        water_acceleration = np.asarray([wax, way, waz])

        return water_velocity, water_acceleration

    def plot_ocean(self, x_range, y_range, x_num, y_num, time, ax):

        nx, ny = (x_num, y_num)
        x = np.linspace( x_range[0], x_range[1], nx)
        y = np.linspace( y_range[0], y_range[1], ny)
        xv, yv = np.meshgrid(x, y)


        eta = np.zeros((ny,nx))

        for i in range(ny):
            for j in range(nx):
                self.cal_wave_field(xv[i,j], yv[i,j], 0, time, 0)
                eta[i,j] = self.eta

        ax.plot_surface(xv,yv,eta, linewidth=0, alpha=0.8,)
        ax.plot_surface(xv,yv,-self.water_depth, linewidth=0)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    oceana = OCEAN("Params.json")


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    oceana.plot_ocean([-5,30], [-15,15], 5, 5, 0, ax)
    oceana.cal_wave_field(0,1,2,0)
    print (oceana.eta)
    plt.show()
    print ('=============================================')
    oceana.plot_ocean([-5,30], [-15,15], 5, 5, 1, ax)
    print (oceana.cal_wave_field(0,1,-0.5,5))
    plt.show()