import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from astropy.coordinates import EarthLocation
from astropy import coordinates as coord
import scipy.fft as fft

from image_code import image_calculation_functions as ic


def generate_ps(datacube,dU):                   #Data is a 2D array and binwidth is a floating point

    print("Averaging")    
    y, x = np.indices(datacube.shape)                                      #X & Y indices of the array
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)               #Distance from the center
    bin_width = 1.0
    bins = np.arange(0,np.max(r),bin_width)
    Nbins = len(bins)        
    radial_prof = np.zeros((Nbins),dtype='float32')
    UU = np.zeros((Nbins))
    for i in range(Nbins):
        r_min = bins[i] 
        r_max = bins[i] + bin_width
        idx = (r >= r_min) * (r < r_max)
        radial_prof[i] = np.mean(datacube[idx])
        UU[i] = r_min#np.mean(r[idx])
                    
    return UU*dU, radial_prof

#pauli matrices
sig_0 = np.array([[1,0],[0,1]])
sig_2 = np.array([[0,1],[1,0]])
sig_3 = np.array([[0,1j],[-1j,0]])
sig_1 = np.array([[1,0],[0,-1]])

def uv_to_psf(u,v,factor):
    bins = factor
    r_b = bins/2
    H, xedges, yedges = np.histogram2d((u.flatten()),(v.flatten()),bins=(bins,bins),normed = True, range=[[-r_b,r_b],[-r_b,r_b]])
   
    #X, Y = np.meshgrid(xedges, yedges)
    theta_x = fft.fftshift(fft.fftfreq(bins,1/(np.pi)))*180/np.pi
    theta_y = fft.fftshift(fft.fftfreq(bins,1/(np.pi)))*180/np.pi
    t_x,t_y = np.meshgrid(theta_x,theta_y)

    H1_ft = (fft.ifftshift(fft.ifft2(fft.ifftshift(H)))) #* (bins/2)**2
    H1_ft = H1_ft/np.max(H1_ft)
    
    return t_x,t_y, H1_ft

def muller_cal(etheta_square,ephi_square,etheta_square90,ephi_square90,del_u,del_v,offset=0):
    M00 = np.zeros_like(etheta_square,dtype='complex')
    M01 = np.zeros_like(etheta_square,dtype='complex')
    M02 = np.zeros_like(etheta_square,dtype='complex')
    M03 = np.zeros_like(etheta_square,dtype='complex')

    M10 = np.zeros_like(etheta_square,dtype='complex')
    M11 = np.zeros_like(etheta_square,dtype='complex')
    M12 = np.zeros_like(etheta_square,dtype='complex')
    M13 = np.zeros_like(etheta_square,dtype='complex')

    M20 = np.zeros_like(etheta_square,dtype='complex')
    M21 = np.zeros_like(etheta_square,dtype='complex')
    M22 = np.zeros_like(etheta_square,dtype='complex')
    M23 = np.zeros_like(etheta_square,dtype='complex')

    M30 = np.zeros_like(etheta_square,dtype='complex')
    M31 = np.zeros_like(etheta_square,dtype='complex')
    M32 = np.zeros_like(etheta_square,dtype='complex')
    M33 = np.zeros_like(etheta_square,dtype='complex')

    phi = np.linspace(0,360,361)*np.pi/180
    theta = np.linspace(0,90,91)*np.pi/180
    
    
    for i in range(len(etheta_square[:90,0])):
        for j in range(len(etheta_square[0,:])):


            l = np.sin(theta[i])*np.cos(phi[j])
            m = np.sin(theta[i]) * np.sin(phi[j])                             
            del_phi = 2*np.pi* (del_u*l + del_v*m)
            J_b = np.array([[1,0],[0,np.exp(1j*del_phi)]])

            J = np.array([[etheta_square[i,j],ephi_square[i,j]],[etheta_square90[i,j],ephi_square90[i,j]]])
            if offset==1:
                J = np.dot(J_b,J)
            if offset ==2:
                J = J_b
            M00[i,j] = np.trace(np.dot(sig_0 , J).dot(sig_0).dot(np.conj(J.T)))
            M01[i,j] = np.trace(np.dot(sig_0 , J).dot(sig_1).dot(np.conj(J.T)))
            M02[i,j] = np.trace(np.dot(sig_0 , J).dot(sig_2).dot(np.conj(J.T)))
            M03[i,j] = np.trace(np.dot(sig_0 , J).dot(sig_3).dot(np.conjugate(J.T)))

            M10[i,j] = np.trace(np.dot(sig_1, J).dot(sig_0).dot(np.conjugate(J.T)))
            M11[i,j] = np.trace(np.dot(sig_1, J).dot(sig_1).dot(np.conjugate(J.T)))
            M12[i,j] = np.trace(np.dot(sig_1, J).dot(sig_2).dot(np.conjugate(J.T)))
            M13[i,j] = np.trace(np.dot(sig_1, J).dot(sig_3).dot(np.conjugate(J.T)))

            M20[i,j] = np.trace(np.dot(sig_2, J).dot(sig_0).dot(np.conjugate(J.T)))
            M21[i,j] = np.trace(np.dot(sig_2, J).dot(sig_1).dot(np.conjugate(J.T)))
            M22[i,j] = np.trace(np.dot(sig_2, J).dot(sig_2).dot(np.conjugate(J.T)))
            M23[i,j] = np.trace(np.dot(sig_2, J).dot(sig_3).dot(np.conjugate(J.T)))

            M30[i,j] = np.trace(np.dot(sig_3, J).dot(sig_0).dot(np.conjugate(J.T)))
            M31[i,j] = np.trace(np.dot(sig_3, J).dot(sig_1).dot(np.conjugate(J.T)))
            M32[i,j] = np.trace(np.dot(sig_3, J).dot(sig_2).dot(np.conjugate(J.T)))
            M33[i,j] = np.trace(np.dot(sig_3, J).dot(sig_3).dot(np.conjugate(J.T)))


    
            
    return M00,M01,M02,M03,M10,M11,M12,M13,M20,M21,M22,M23,M30,M31,M32,M33

def muller_cal_s(etheta_square,ephi_square,etheta_square90,ephi_square90,del_u,del_v,offset=0):
    M = np.zeros((4,4, np.shape(etheta_square)[0], np.shape(etheta_square)[1]),dtype='complex')
    
    phi = np.linspace(0,360,361)*np.pi/180
    theta = np.linspace(0,90,91)*np.pi/180
    
    
    for i in range(len(theta)):
        for j in range(len(phi)):


            l = np.sin(theta[i])*np.cos(phi[j])
            m = np.sin(theta[i]) * np.sin(phi[j])                             
            del_phi = 2*np.pi* (del_u*l + del_v*m)
            J_b = np.array([[1,0],[0,np.exp(1j*del_phi)]])

            J = np.array([[etheta_square[i,j],ephi_square[i,j]],[etheta_square90[i,j],ephi_square90[i,j]]])
            if offset==1:
                J = np.dot(J_b,J)
            if offset ==2:
                J = J_b
            
            M[:,:,i,j] = 0.5 * np.dot(np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,-1j,1j,0]]) , np.kron(J,np.conj(J))).dot(np.array([[1,1,0,0],[0,0,1,1j],[0,0,1,-1j],[1,-1,0,0]]))
          
            
    return M[0,0],M[0,1],M[0,2],M[0,3],M[1,0],M[1,1],M[1,2],M[1,3],M[2,0],M[2,1],M[2,2],M[2,3],M[3,0],M[3,1],M[3,2],M[3,3]

def plot_muller_cal_real(M_matrix, title = 'dipole',save_fig=True):
    phi = np.linspace(0,360,361)*np.pi/180
    theta = np.linspace(0,90,91)*np.pi/180
    
    n_M00 = np.max(np.real(M_matrix[0]))
    lth=1e-4
    P,T = np.meshgrid(phi,theta)
  
    plt.rc('font',size=7)
    plt.rc('axes', labelsize=7)
    
    fig,ax = plt.subplots(4,4, subplot_kw=dict(polar=True),figsize=(7.5,5))
    plt.subplots_adjust(hspace=0.2,wspace=0.35)
    
    p = ax[0,0].pcolormesh(P,T,np.real(M_matrix[0][:90,:])/n_M00,cmap='coolwarm',vmax=1,vmin=1e-2, norm = SymLogNorm(linthresh=lth), rasterized=True)#,  norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,0].set_title('M00',{'fontsize':8},pad=0.0)
    ax[0,0].set_yticklabels([])
    ax[0,0].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[0,0].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[0,0],pad=0.1,shrink=0.8,ticks=[1e-2,1e-1,1],format= '% 1.0e')
    cbar.ax.set_yticklabels(['1e-2', '1e-1', ' 1']) 

    p = ax[0,1].pcolormesh(P,T,np.real(M_matrix[1][:90,:])/n_M00,cmap='coolwarm',vmin = -1e-2, vmax =1e-2,norm = SymLogNorm(linthresh=lth, ))#, norm = SymLogNorm(linthresh= lth,vmin = -0.01, vmax = 0.01))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,1].set_title('M01',{'fontsize':8},pad=0.0)
    ax[0,1].set_yticklabels([])
    ax[0,1].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[0,1].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[0,1],pad=0.1,shrink=0.8, ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p= ax[0,2].pcolormesh(P,T,np.real(M_matrix[2][:90,:])/n_M00,cmap='coolwarm',vmin = -1e-2, vmax =1e-2,norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax = 1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,2].set_title('M02',{'fontsize':8},pad=0.0)
    ax[0,2].set_yticklabels([])
    ax[0,2].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[0,2].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[0,2],pad=0.1,shrink=0.8,ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[0,3].pcolormesh(P,T,np.real(M_matrix[3][:90,:])/n_M00,cmap='coolwarm',vmin = -1e-2, vmax =1e-2, norm = SymLogNorm(linthresh= lth))#, vmin = -1e-3, vmax = 1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,3].set_title('M03',{'fontsize':8},pad=0.0)
    ax[0,3].set_yticklabels([])
    ax[0,3].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[0,3].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[0,3],pad=0.1,shrink=0.8, ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    ##################################
    p = ax[1,0].pcolormesh(P,T,np.real(M_matrix[4][:90,:])/n_M00,cmap='coolwarm',vmin = -1e-2, vmax =1e-2, norm = SymLogNorm(linthresh= lth))#, vmin = -1e-3, vmax = 1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[1,0].set_title('M10',{'fontsize':8},pad=0.0)
    ax[1,0].set_yticklabels([])
    ax[1,0].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[1,0].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[1,0],pad=0.1,shrink=0.8,ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[1,1].pcolormesh(P,T,np.real(M_matrix[5][:90,:])/n_M00,cmap='coolwarm',vmin = -1, vmax =1, norm = SymLogNorm(linthresh=1e-2, vmin = -1, vmax =1)) #X,Y & data2D must all be same dimensions
    ax[1,1].set_title('M11',{'fontsize':8},pad=0.0)
    ax[1,1].set_yticklabels([])
    ax[1,1].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[1,1].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[1,1],pad=0.1,shrink=0.8,ticks=[-1,-1e-2, 1e-2,1], format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1', '-1e-2', ' 1e-2',' 1']) 

    p=ax[1,2].pcolormesh(P,T,np.real(M_matrix[6][:90,:])/n_M00,cmap='coolwarm',vmin = -1, vmax =1, norm = SymLogNorm(linthresh=1e-2, vmin = -1, vmax=1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[1,2].set_title('M12',{'fontsize':8},pad=0.0)
    ax[1,2].set_yticklabels([])
    ax[1,2].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[1,2].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[1,2],pad=0.1,shrink=0.8, ticks=[-1,-1e-2, 1e-2,1],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1', '-1e-2', ' 1e-2',' 1']) 

    p = ax[1,3].pcolormesh(P,T,np.real(M_matrix[7][:90,:])/n_M00,cmap='coolwarm',vmin = -1e-2, vmax =1e-2, norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[1,3].set_title('M13',{'fontsize':8},pad=0.0)
    ax[1,3].set_yticklabels([])
    ax[1,3].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[1,3].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[1,3],pad=0.1,shrink=0.8, ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    ##################################
    p=ax[2,0].pcolormesh(P,T,np.real(M_matrix[8][:90,:])/n_M00,cmap='coolwarm',vmin = -1e-2, vmax =1e-2,norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,0].set_title('M20',{'fontsize':8},pad=0.0)
    ax[2,0].set_yticklabels([])
    ax[2,0].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[2,0].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[2,0],pad=0.1,shrink=0.8, ticks=[-1e-2,-1e-4,1e-4,1e-2],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[2,1].pcolormesh(P,T,np.real(M_matrix[9][:90,:])/n_M00,cmap='coolwarm',vmin = -1, vmax =1,norm = SymLogNorm(linthresh=1e-2, vmin = -1, vmax=1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,1].set_title('M21',{'fontsize':8},pad=0.0)
    ax[2,1].set_yticklabels([])
    ax[2,1].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[2,1].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[2,1],pad=0.1,shrink=0.8, ticks=[-1,-1e-2, 1e-2,1],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1', '-1e-2', ' 1e-2',' 1']) 

    p = ax[2,2].pcolormesh(P,T,np.real(M_matrix[10][:90,:])/n_M00,cmap='coolwarm',vmin = -1, vmax =1,norm = SymLogNorm(linthresh=1e-2, vmin = -1, vmax =1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,2].set_title('M22',{'fontsize':8},pad=0.0)
    ax[2,2].set_yticklabels([])
    ax[2,2].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[2,2].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[2,2],pad=0.1,shrink=0.8, ticks=[-1,-1e-2, 1e-2,1],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1', '-1e-2', ' 1e-2',' 1']) 

    p = ax[2,3].pcolormesh(P,T,np.real(M_matrix[11][:90,:])/n_M00,cmap='coolwarm',vmin = -1e-2, vmax =1e-2,norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,3].set_title('M23',{'fontsize':8},pad=0.0)
    ax[2,3].set_yticklabels([])
    ax[2,3].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[2,3].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[2,3],pad=0.1,shrink=0.8, ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    ##################################
    p = ax[3,0].pcolormesh(P,T,np.real(M_matrix[12][:90,:])/n_M00,cmap='coolwarm',vmin = -1e-2, vmax =1e-2, norm=SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,0].set_title('M30',{'fontsize':8},pad=0.0)
    ax[3,0].set_yticklabels([])
    ax[3,0].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[3,0].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[3,0],pad=0.1,shrink=0.8, ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[3,1].pcolormesh(P,T,np.real(M_matrix[13][:90,:])/n_M00,cmap='coolwarm',vmin = -1e-2, vmax =1e-2,norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,1].set_title('M31',{'fontsize':8},pad=0.0)
    ax[3,1].set_yticklabels([])
    ax[3,1].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[3,1].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[3,1],pad=0.1,shrink=0.8,ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[3,2].pcolormesh(P,T,np.real(M_matrix[14][:90,:])/n_M00,cmap='coolwarm',vmin = -1e-2, vmax =1e-2, norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,2].set_title('M32',{'fontsize':8},pad=0.0)
    ax[3,2].set_yticklabels([])
    ax[3,2].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[3,2].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[3,2],pad=0.1,shrink=0.8,ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[3,3].pcolormesh(P,T,np.real(M_matrix[15][:90,:]/n_M00),cmap='coolwarm',vmin = -0.5, vmax =0.5, norm = SymLogNorm(linthresh=1e-2))#, vmin = 0, vmax =1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,3].set_title('M33',{'fontsize':8},pad=0.0)
    ax[3,3].set_yticklabels([])
    ax[3,3].set_xticks(np.array([45,135,225,315])*np.pi/180)
    ax[3,3].tick_params(axis='x',pad=-0.5)
    cbar = plt.colorbar(p,ax=ax[3,3],pad=0.1,shrink=0.8,ticks=[1e-2,1e-1,1],format= '% 1.0e')
    cbar.ax.set_yticklabels(['1e-2', '1e-1', '1']) 
    
    if save_fig==True:
        plt.savefig('/data4/nmahesh/edges/Lunar/plots/'+title+'.pdf',dpi=300,bbox_inches = 'tight')

def plot_muller_cal_abs(M_matrix,title='dipole',save_fig=False):

    phi = np.linspace(0,360,361)*np.pi/180
    theta = np.linspace(0,90,91)*np.pi/180
    
               
    
    n_M00 = np.max(np.real(M_matrix[0]))
    lth=1e-4
    P,T = np.meshgrid(phi,theta)
  
    plt.rc('font',size=7)
    plt.rc('axes', labelsize=7)
    
    fig,ax = plt.subplots(4,4, subplot_kw=dict(polar=True),figsize=(7.5,5))#,constrained_layout=True)
    plt.subplots_adjust(hspace=0.2,wspace=0.35)
    p = ax[0,0].pcolormesh(P,T,np.abs(M_matrix[0][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmax=1,vmin=1e-2, norm = SymLogNorm(linthresh=lth), rasterized=True)#,  norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,0].set_title('M00',{'fontsize':8},pad=0.0)
    ax[0,0].set_yticklabels([])
    ax[0,0].set_xticks([])
    ax[0,0].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[0,0].tick_params(axis='x',pad=-0.5)
    ax[0,0].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[0,0],pad=0.1,shrink=0.8)#,ticks=[1e-2,1e-1,1],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['1e-2', '1e-1', ' 1']) 

    p = ax[0,1].pcolormesh(P,T,np.abs(M_matrix[1][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1e-2, vmax =1e-2,norm = SymLogNorm(linthresh=lth, ))#, norm = SymLogNorm(linthresh= lth,vmin = -0.01, vmax = 0.01))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,1].set_title('M01',{'fontsize':8},pad=0.0)
    ax[0,1].set_yticklabels([])
    ax[0,1].set_xticks([])
    ax[0,1].set_rgrids([np.pi/6,np.pi/3],[30,60])  
    ax[0,1].tick_params(axis='x',pad=-0.5)
    ax[0,1].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[0,1],pad=0.1,shrink=0.8)#, ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p= ax[0,2].pcolormesh(P,T,np.abs(M_matrix[2][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1e-2, vmax =1e-2,norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax = 1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,2].set_title('M02',{'fontsize':8},pad=0.0)
    ax[0,2].set_yticklabels([])
    ax[0,2].set_xticks([])
    ax[0,2].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[0,2].tick_params(axis='x',pad=-0.5)
    ax[0,2].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[0,2],pad=0.1,shrink=0.8)#,ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[0,3].pcolormesh(P,T,np.abs(M_matrix[3][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1e-2, vmax =1e-2, norm = SymLogNorm(linthresh= lth))#, vmin = -1e-3, vmax = 1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,3].set_title('M03',{'fontsize':8},pad=0.0)
    ax[0,3].set_yticklabels([])
    ax[0,3].set_xticks([])
    ax[0,3].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[0,3].tick_params(axis='x',pad=-0.5)
    ax[0,3].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[0,3],pad=0.1,shrink=0.8)#, ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    ##################################
    p = ax[1,0].pcolormesh(P,T,np.abs(M_matrix[4][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1e-2, vmax =1e-2, norm = SymLogNorm(linthresh= lth))#, vmin = -1e-3, vmax = 1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[1,0].set_title('M10',{'fontsize':8},pad=0.0)
    ax[1,0].set_yticklabels([])
    ax[1,0].set_xticks([])
    ax[1,0].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[1,0].tick_params(axis='x',pad=-0.5)
    ax[1,0].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[1,0],pad=0.1,shrink=0.8)#,ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[1,1].pcolormesh(P,T,np.abs(M_matrix[5][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1, vmax =1, norm = SymLogNorm(linthresh=1e-2, vmin = -1, vmax =1)) #X,Y & data2D must all be same dimensions
    ax[1,1].set_title('M11',{'fontsize':8},pad=0.0)
    ax[1,1].set_yticklabels([])
    ax[1,1].set_xticks([])
    ax[1,1].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[1,1].tick_params(axis='x',pad=-0.5)
    ax[1,1].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[1,1],pad=0.1,shrink=0.8)#,ticks=[-1,-1e-2, 1e-2,1], format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1', '-1e-2', ' 1e-2',' 1']) 

    p=ax[1,2].pcolormesh(P,T,np.abs(M_matrix[6][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1, vmax =1, norm = SymLogNorm(linthresh=1e-2, vmin = -1, vmax=1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[1,2].set_title('M12',{'fontsize':8},pad=0.0)
    ax[1,2].set_yticklabels([])
    ax[1,2].set_xticks([])
    ax[1,2].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[1,2].tick_params(axis='x',pad=-0.5)
    ax[1,2].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[1,2],pad=0.1,shrink=0.8)#, ticks=[-1,-1e-2, 1e-2,1],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1', '-1e-2', ' 1e-2',' 1']) 

    p = ax[1,3].pcolormesh(P,T,np.abs(M_matrix[7][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1e-2, vmax =1e-2, norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[1,3].set_title('M13',{'fontsize':8},pad=0.0)
    ax[1,3].set_yticklabels([])
    ax[1,3].set_xticks([])
    ax[1,3].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[1,3].tick_params(axis='x',pad=-0.5)
    ax[1,3].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[1,3],pad=0.1,shrink=0.8)#, ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    ##################################
    p=ax[2,0].pcolormesh(P,T,np.abs(M_matrix[8][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1e-2, vmax =1e-2,norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,0].set_title('M20',{'fontsize':8},pad=0.0)
    ax[2,0].set_yticklabels([])
    ax[2,0].set_xticks([])
    ax[2,0].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[2,0].tick_params(axis='x',pad=-0.5)
    ax[2,0].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[2,0],pad=0.1,shrink=0.8)#, ticks=[-1e-2,-1e-4,1e-4,1e-2],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[2,1].pcolormesh(P,T,np.abs(M_matrix[9][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1, vmax =1,norm = SymLogNorm(linthresh=1e-2, vmin = -1, vmax=1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,1].set_title('M21',{'fontsize':8},pad=0.0)
    ax[2,1].set_yticklabels([])
    ax[2,1].set_xticks([])
    ax[2,1].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[2,1].tick_params(axis='x',pad=-0.5)
    ax[2,1].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[2,1],pad=0.1,shrink=0.8)#, ticks=[-1,-1e-2, 1e-2,1],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1', '-1e-2', ' 1e-2',' 1']) 

    p = ax[2,2].pcolormesh(P,T,np.abs(M_matrix[10][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1, vmax =1,norm = SymLogNorm(linthresh=1e-2, vmin = -1, vmax =1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,2].set_title('M22',{'fontsize':8},pad=0.0)
    ax[2,2].set_yticklabels([])
    ax[2,2].set_xticks([])
    ax[2,2].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[2,2].tick_params(axis='x',pad=-0.5)
    ax[2,2].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[2,2],pad=0.1,shrink=0.8)#, ticks=[-1,-1e-2, 1e-2,1],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1', '-1e-2', ' 1e-2',' 1']) 

    p = ax[2,3].pcolormesh(P,T,np.abs(M_matrix[11][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1e-2, vmax =1e-2,norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,3].set_title('M23',{'fontsize':8},pad=0.0)
    ax[2,3].set_yticklabels([])
    ax[2,3].set_xticks([])
    ax[2,3].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[2,3].tick_params(axis='x',pad=-0.5)
    ax[2,3].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[2,3],pad=0.1,shrink=0.8)#, ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    ##################################
    p = ax[3,0].pcolormesh(P,T,np.abs(M_matrix[12][:90,:])/n_M00,cmap='coolwarm')#,vmin = -1e-2, vmax =1e-2, norm=SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,0].set_title('M30',{'fontsize':8},pad=0.0)
    ax[3,0].set_yticklabels([])
    ax[3,0].set_xticks([])
    ax[3,0].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[3,0].tick_params(axis='x',pad=-0.5)
    ax[3,0].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[3,0],pad=0.1,shrink=0.8)#, ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[3,1].pcolormesh(P,T,np.abs(M_matrix[13][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1e-2, vmax =1e-2,norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,1].set_title('M31',{'fontsize':8},pad=0.0)
    ax[3,1].set_yticklabels([])
    ax[3,1].set_xticks([])
    ax[3,1].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[3,1].tick_params(axis='x',pad=-0.5)
    ax[3,1].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[3,1],pad=0.1,shrink=0.8)#,ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[3,2].pcolormesh(P,T,np.abs(M_matrix[14][:90,:])/n_M00,cmap='coolwarm',rasterized=True)#,vmin = -1e-2, vmax =1e-2, norm = SymLogNorm(linthresh=lth))#, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,2].set_title('M32',{'fontsize':8},pad=0.0)
    ax[3,2].set_yticklabels([])
    ax[3,2].set_xticks([])
    ax[3,2].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[3,2].tick_params(axis='x',pad=-0.5)
    ax[3,2].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[3,2],pad=0.1,shrink=0.8)#,ticks=[-1e-2,-1e-4, 1e-4,1e-2],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['-1e-2', '-1e-4', ' 1e-4',' 1e-2']) 

    p = ax[3,3].pcolormesh(P,T,np.abs(M_matrix[15][:90,:]/n_M00),cmap='coolwarm',rasterized=True)#,vmin = 1e-2, vmax =1, norm = SymLogNorm(linthresh=1e-6))#, vmin = 0, vmax =1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,3].set_title('M33',{'fontsize':8},pad=0.0)
    ax[3,3].set_yticklabels([])
    ax[3,3].set_xticks([])
    ax[3,3].set_rgrids([np.pi/6,np.pi/3],[30,60])
    ax[3,3].tick_params(axis='x',pad=-0.5)
    ax[3,3].grid(linewidth=1,color='k')
    cbar = plt.colorbar(p,ax=ax[3,3],pad=0.1,shrink=0.8)#,ticks=[1e-2,1e-1,1],format= '% 1.0e')
    #cbar.ax.set_yticklabels(['1e-2', '1e-1', '1']) 
    
    if save_fig==True:
        plt.savefig('/data4/nmahesh/edges/Lunar/plots/'+title+'.pdf',dpi=300,bbox_inches = 'tight')




    

def offset_vmuller_cal(de,H_o,off):
    
    phi = np.linspace(0,360,361)*np.pi/180
    theta = np.linspace(0,90,91)*np.pi/180
    wav = np.array([500,150,30])
    freq = 300/wav
    
    lth=1e-4
    
    P,T = np.meshgrid(phi,theta)
   
    plt.rc('font',size=7)
    plt.rc('axes', labelsize=7)
    for k in range(3):   
        fig,ax = plt.subplots(2,2, subplot_kw=dict(polar=True),figsize=(4,3.5))#,constrained_layout=True)
        M = np.zeros((4,4,len(theta),len(phi)),dtype='complex')
        for i in range(len(theta)):
            for j in range(len(phi)):

                l = np.sin(theta[i])*np.cos(phi[j])
                m = np.sin(theta[i]) * np.sin(phi[j])  
                
                del_u,del_v,del_w = ic.uvcal(de,H_o,wav[k],off,off)
                del_phi = 2*np.pi* (del_u*l + del_v*m)
                
                J= np.array([[1,0],[0,np.exp(1j*del_phi)]])

                M[:,:,i,j] = 0.5 * np.dot(np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,-1j,1j,0]]) , np.kron(J,np.conj(J))).dot(np.array([[1,1,0,0],[0,0,1,1j],[0,0,1,-1j],[1,-1,0,0]]))
          

               

       

        ##################################
        p = ax[0,0].pcolormesh(P,T,np.abs(M[2,2,:90,:]),cmap='coolwarm', vmin = 0, vmax =1,rasterized=True)# norm=SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
        ax[0,0].set_title('M22',{'fontsize':8},pad=0.0)
        ax[0,0].set_yticklabels([])
        ax[0,0].set_xticks([])
        ax[0,0].set_rgrids([np.pi/4,np.pi/2],[45,90])
        ax[0,0].tick_params(axis='x',pad=-0.5)
        ax[0,0].grid(linewidth=1,color='k')
        plt.colorbar(p,ax=ax[0,0],pad=0.1,shrink=0.8,ticks=[0,0.5,1.0])

        p = ax[0,1].pcolormesh(P,T,np.abs(M[2,3,:90,:]),cmap='coolwarm',vmin = 0, vmax =1,rasterized=True)#norm = SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
        ax[0,1].set_title('M23',{'fontsize':8},pad=0.0)
        ax[0,1].set_yticklabels([])
        ax[0,1].set_xticks([])
        ax[0,1].set_rgrids([np.pi/4,np.pi/2],[45,90])
        ax[0,1].tick_params(axis='x',pad=-0.5)
        ax[0,1].grid(linewidth=1,color='k')
        plt.colorbar(p,ax=ax[0,1],pad=0.1,shrink=0.8,ticks=[0,0.5,1.0])

        p = ax[1,0].pcolormesh(P,T,np.abs(M[3,2,:90,:]),cmap='coolwarm',vmin = 0, vmax =1,rasterized=True)#norm = SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
        ax[1,0].set_title('M32',{'fontsize':8},pad=0.0)
        ax[1,0].set_yticklabels([])
        ax[1,0].set_xticks([])
        ax[1,0].set_rgrids([np.pi/4,np.pi/2],[45,90])
        ax[1,0].tick_params(axis='x',pad=-0.5)
        ax[1,0].grid(linewidth=1,color='k')
        plt.colorbar(p,ax=ax[1,0],pad=0.1,shrink=0.8,ticks=[0,0.5,1.0])

        p = ax[1,1].pcolormesh(P,T,np.abs(M[3,3,:90,:]),cmap='coolwarm',vmin = 0, vmax =1,rasterized=True)#norm = SymLogNorm(linthresh=lth, vmin = 0, vmax =1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
        ax[1,1].set_title('M33',{'fontsize':8},pad=0.0)
        ax[1,1].set_yticklabels([])
        ax[1,1].set_xticks([])
        ax[1,1].set_rgrids([np.pi/4,np.pi/2],[45,90])
        ax[1,1].tick_params(axis='x',pad=-0.5)
        ax[1,1].grid(linewidth=1,color='k')
        plt.colorbar(p,ax=ax[1,1],pad=0.1,shrink=0.8,ticks=[0,0.5,1.0])

        plt.savefig('/data4/nmahesh/edges/Lunar/plots/muller_only_offset_'+str(int(freq[k]))+'.pdf',dpi=200,bbox_inches = 'tight')
    
        
        

   
dipolehalf = 0.050          #meters
pattern = 'log'
rovers = 4

if pattern == 'log':
    
    if rovers == 1:
        a = 5 # starting radius
        b = 0.12 # eccentricity
        w = 10 #num wraps
        rot = [0]
    elif rovers == 2:
        a = 5 
        b = 0.19 
        w = 6
        rot = [0,np.pi]
    elif rovers == 3:
        a = 5 
        b = 0.282 
        w = 4
        rot = [0, 2/3*np.pi, 4/3*np.pi]
    elif rovers == 4:
        a = 5 
        b = 0.565 
        w = 2
        rot = [0, np.pi/2, np.pi, 3/2*np.pi]

elif pattern == 'arch':
    
    if rovers == 1:
        a = 250 #spacing
        w = 4 # wraps
        rot = [0]
    elif rovers == 2:
        a = 250
        w = 4
        rot = [0,np.pi]
    elif rovers == 3:
        a = 250
        w = 4
        rot = [0, 2/3*np.pi, 4/3*np.pi]
    elif rovers == 4:
        a = 250
        w = 4
        rot = [0, np.pi/2, np.pi, 3/2*np.pi]

elif pattern == 'petal':
    a = 5e3
    w = 1


r=[]
theta = []
tsteps = np.linspace(0,w*2*np.pi,1000)

for t in tsteps:
    theta.append(t)
    #radius of spiral (polar coords)
    if pattern == 'log':
        r.append(a*np.exp(b*t))
    elif pattern == 'arch':
        r.append(a*t)
    elif pattern == 'petal':
        r.append(a*np.cos(2*t))
        

x = r*np.cos(theta)*1e-3
y = r*np.sin(theta)*1e-3

""" recenter"""
#x=x-x[0]
#y=y-y[0]
'''
plt.plot(x,y,"--b")
plt.gca().set_aspect('equal')
plt.grid()
plt.xlabel('km')
plt.ylabel('km')
plt.show()

'''

""" now let's find node positions 

for now, evenly spaced along the spiral (since the "closer together in the middle" is 
accomplished by the spiral)
"""
Nnodes = 32
dx = np.diff(x)
dy = np.diff(y)
dd = np.sqrt(dx**2+dy**2)
cumdist = np.cumsum(dd)
totallen = cumdist[-1]
#print("Total length: %f"%totallen)

spacing= totallen/float(Nnodes)
#print("notional spacing: %f"%spacing)

notional=[]
nextpoint = spacing
for idx,d in enumerate(cumdist):
    if d>nextpoint:
        #print("%d %f %f"%(idx,x[idx],y[idx]))
        notional.append(idx)
        nextpoint = nextpoint + spacing
""" add the end """
notional.append(len(cumdist)-1)

#print("notional positions")
#for idx in notional:
    #print("%d %f %f"%(idx,x[idx],y[idx]))
    #plt.plot(x[idx],y[idx],'.m')

""" build the track 

there are 4 possible configurations of the two 100 m segments:
          _    _
    _| |_  |  |

and we choose based on the local slope of the trajectory
"""

track=[]
track.append([0,0])

xnode=[]

ynode=[]



for idx in notional:
    cx = x[idx]
    cy = y[idx]
    cxp = cx+dipolehalf
    cxm = cx-dipolehalf
    cyp = cy+dipolehalf
    cym = cy-dipolehalf
    """ these rules need some work """
    if dx[idx]>0:
        if dy[idx]>0:
            """ do x direction first, then y"""
            track.append([cxm,cym])
            track.append([cx,cym])
            track.append([cxp,cym])
            xnode.append([cx,cym])

            track.append([cxp,cym])     #dupes previous.. eventually insert a space here
            track.append([cxp,cy])
            track.append([cxp,cyp])
            ynode.append([cxp,cy])
            
        else:
            """ do Y first then x"""
            """
            track.append([cxm,cym])
            track.append([cxm,cy])
            track.append([cxm,cyp])
            ynode.append([cxm,cy])

            track.append([cxm,cyp])     #dupes previous.. eventually insert a space here
            track.append([cx,cyp])
            track.append([cxp,cyp])
            xnode.append([cx,cyp])
            """
            track.append([cxm,cyp])
            track.append([cx,cyp])
            track.append([cxp,cyp])
            xnode.append([cx,cyp])

            track.append([cxp,cyp])     #dupes previous.. eventually insert a space here
            track.append([cxp,cy])
            track.append([cxp,cym])
            ynode.append([cxp,cy])

    else:
        if dy[idx]>0:
            track.append([cxp,cym])
            track.append([cx,cym])
            track.append([cxm,cym])
            xnode.append([cx,cym])

            track.append([cxm,cym])     #dupes previous.. eventually insert a space here
            track.append([cxm,cy])
            track.append([cxm,cyp])
            ynode.append([cxm,cy])

        else:
            track.append([cxp,cyp])
            track.append([cx,cyp])
            track.append([cxm,cyp])
            xnode.append([cx,cyp])

            track.append([cxm,cyp])     #dupes previous.. eventually insert a space here
            track.append([cxm,cy])
            track.append([cxm,cym])
            ynode.append([cxm,cy])


""" and let's see if it plots """
'''
px,py = zip(*track)
plt.plot(px,py)
    
xx,xy= zip(*xnode)
yx,yy = zip(*ynode)

plt.plot(xx,xy,'.r')
plt.plot(yx,yy,'.g')

pxd = np.diff(px)
pyd = np.diff(py)
pd = np.cumsum(np.sqrt(pxd**2+pyd**2))

txt = "basis curve distance: %5.2f km\nZigzag distance %5.2f km"%(cumdist[-1],pd[-1])
#plt.text(2,0,txt)
'''