import numpy as np
from astropy.coordinates import EarthLocation
from astropy import coordinates as coord
from scipy.interpolate import RectBivariateSpline
from itertools import permutations
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import scipy.fft as fft
from astropy.time import Time
from matplotlib.colors import LogNorm

def lmtotp(L,M):
    """
    converts from L and M meshgrid to corresponding theta and phi meshgrid
    """
    q = L**2+M**2
    el = np.sqrt(1 - q)
    if q>=1:
        el = 0

    p = -np.arctan2(M,L)#-np.arctan2(M,L)#
    t = np.pi/2 - np.arcsin(el)#-np.arcsin(L,(np.sin(p)))+np.pi/2
    return p,t


def beamtopol(beam_pol):
    """
    Calculate Stokes I, Q, U and V images given XX, YY, XY and YX images
    """
    sky_pol = np.zeros_like(beam_pol,dtype=np.complex128)
    sky_pol[0] = (beam_pol[0] + beam_pol[3])/2
    sky_pol[1] = (beam_pol[0] - beam_pol[3])/2
    sky_pol[2] = (beam_pol[1] + beam_pol[2])/2
    sky_pol[3] = (-1j*beam_pol[1] + 1j* beam_pol[2])/2
    
    return sky_pol


def beamtopol_real(beam_pol):
    """
    Calculate Stokes I, Q, U and V images from real components of XX, YY, XY and YX images
    """
    sky_pol = np.zeros_like(beam_pol,dtype=np.complex128)
    sky_pol[0] = (np.real(beam_pol[0]) + np.real(beam_pol[3]))/2
    sky_pol[1] = (np.real(beam_pol[0]) - np.real(beam_pol[3]))/2
    sky_pol[2] = (np.real(beam_pol[1]) + np.real(beam_pol[2]))/2
    sky_pol[3] = (-1j*np.real(beam_pol[1]) + 1j* np.real(beam_pol[2]))/2
    
    return sky_pol


def altaz_sources(coos,times):
    """
    Transform the source coordinates from the absolute sky frame to the antenna frame.
    """
      
    farside_loc = EarthLocation.from_geodetic(lat = 0.0, lon = 180.0)
    aa_frame = coord.AltAz(obstime=times, location=farside_loc)
    
    aa_coos = coos.transform_to(aa_frame)  
    
    return aa_coos.alt.deg, aa_coos.az.deg


def uvcal(dec,H_o,wav,y_off,z_off):
    """
    calculate the u,v,w coverage given the baselines and source position
    """
    x =0
    u = np.sin(H_o)*x+ np.cos(H_o) * y_off/wav
    v = -np.sin(dec)*np.cos(H_o)*x + np.cos(dec)*z_off/wav + np.sin(dec)*np.sin(H_o)*y_off/wav
    w =  np.cos(dec)*np.cos(H_o)*x+ np.sin(dec)*z_off/wav - np.cos(dec)*np.sin(H_o)*y_off/wav
    return u,v,w


def baseline_cal(dec,H_o,wav,far_array):
    """
    Given antenna positions in the array, calculate the baselines and u,v,w coverage
    """
    n_basefa = permutations(np.arange(np.shape(far_array)[0]),2)
    basefa = (list(n_basefa))
    for i in range(np.shape(far_array)[0]):
        basefa.append((i,i))
    
    u = np.zeros((len(basefa)))
    v = np.zeros((len(basefa)))
    w = np.zeros((len(basefa)))
    
    for i in range(len(basefa)): 
                yfa = (far_array[basefa[i][1],0] - far_array[basefa[i][0],0])*10**3/wav
                zfa = (far_array[basefa[i][1],1] - far_array[basefa[i][0],1])*10**3/wav
                
                u[i],v[i],w[i] = uvcal(dec,H_o,wav,yfa,zfa)
    return u,v,w


def beam_interpolate(beam,theta,phi,t,p):
    """
    interpolates the beam from theta and phi to a rectangular grid.
    """
    interp_spline = RectBivariateSpline(theta,phi,np.real(beam[:91,:]))
    beam_new = interp_spline(t,p,grid=False)
    
    interp_spline = RectBivariateSpline(theta,phi,np.imag(beam[:91,:]))
    beam_newi = interp_spline(t,p,grid=False)
    
    beam_interp = beam_new + 1j* beam_newi
    
    return beam_interp


def read_feko(file_name_prefix,freq_p,theta_p,phi_p):
    """
    Reads in a typical feko output file and creates a beam object.
    """
    beam_square = np.zeros((freq_p,theta_p,phi_p))
    gain_theta90 = np.zeros((freq_p,theta_p,phi_p))
    gain_phi90 = np.zeros((freq_p,theta_p,phi_p))
    etheta_square90 = np.zeros((freq_p,theta_p,phi_p),dtype = 'complex')
    ephi_square90 = np.zeros((freq_p,theta_p,phi_p),dtype = 'complex')
   
    f1 = open(file_name_prefix+'_0-90.ffe')
    f2 = open(file_name_prefix+'_91-180.ffe')
    f3 = open(file_name_prefix+'_181-270.ffe')
    f4 = open(file_name_prefix+'_271-360.ffe')

    z = 181*91 +10# ---> change this to no.of theta * no.of phi + No.of header lines
    c=0
    for line in f1:
        if c%z ==0:
            co=0
        if c % z >= 10: 
            x = list(map(float,line.split()))
            beam_square[int(c/z), co%181,int(co/181)] = 10**(x[8]/10)##*** beam_square [freq,theta,phi]***
            etheta_square90[int(c/z), co%181,int(co/181)] = x[2] + 1j*(x[3])
            ephi_square90[int(c/z), co%181,int(co/181)] = x[4] + 1j*x[5]
            gain_theta90[int(c/z), co%181,int(co/181)] = 10**(x[6]/10)
            gain_phi90[int(c/z), co%181,int(co/181)] = 10**(x[7]/10)
            co = co+1
        c = c+1

    z = 181*90 +10#   
    c = 0    
    for line in f2:
        if c%z ==0:
            co=0
        if c % z >= 10: 
            x = list(map(float,line.split()))
            #print(c/z)
            beam_square[int(c/z), co%181,int(co/181)+91] = 10**(x[8]/10)  ##*** beam_square [freq,theta,phi]***
            etheta_square90[int(c/z), co%181,int(co/181)+91] = x[2] + 1j*x[3]
            ephi_square90[int(c/z), co%181,int(co/181)+91] = x[4] + 1j*x[5]
            gain_theta90[int(c/z), co%181,int(co/181)+91] = 10**(x[6]/10)
            gain_phi90[int(c/z), co%181,int(co/181)+91] = 10**(x[7]/10)
            co = co+1
        c = c+1

    c = 0    
    for line in f3:
        if c%z ==0:
            co=0
        if c % z >= 10: 
            x = list(map(float,line.split()))
            beam_square[int(c/z), co%181,int(co/181)+181] = 10**(x[8]/10)  ##*** beam_square [freq,theta,phi]***
            etheta_square90[int(c/z), co%181,int(co/181)+181] = x[2] + 1j*x[3]
            ephi_square90[int(c/z), co%181,int(co/181)+181] = x[4] + 1j*x[5]
            gain_theta90[int(c/z), co%181,int(co/181)+181] = 10**(x[6]/10)
            gain_phi90[int(c/z), co%181,int(co/181)+181] = 10**(x[7]/10)
            co = co+1
        c = c+1


    c = 0    
    for line in f4:
        if c%z ==0:
            co=0
        if c % z >= 10: 
            x = list(map(float,line.split()))
            beam_square[int(c/z), co%181,int(co/181)+271] = 10**(x[8]/10)  ##*** beam_square [freq,theta,phi]***
            etheta_square90[int(c/z), co%181,int(co/181)+271] = x[2] + 1j*x[3]
            ephi_square90[int(c/z), co%181,int(co/181)+271] = x[4] + 1j*x[5]
            gain_theta90[int(c/z), co%181,int(co/181)+271] = 10**(x[6]/10)
            gain_phi90[int(c/z), co%181,int(co/181)+271] = 10**(x[7]/10)
            co = co+1
        c = c+1
    return beam_square, etheta_square90, ephi_square90


class processed_source:
    """
    Creates a source object and processes it through the complete FARSIDE pipeline
    """
    def init_skycoords(self,ra_coords,dec_coords,flux,t):
        
        self.map_objs = SkyCoord(ra=ra_coords*u.degree, dec=dec_coords*u.degree)
        self.map_objs_altaz = altaz_sources(self.map_objs,t)
        
        self.alt = 90 - self.map_objs_altaz[0]
        self.az = self.map_objs_altaz[1]
                
        self.sky_map_tp = np.zeros((len(ra_coords),3))
        for i in range(len(ra_coords)):
            self.sky_map_tp[i,0] = flux[i]
            self.sky_map_tp[i,1] = self.alt[i]
            if self.az[i]>180:
                self.sky_map_tp[i,2] = self.az[i]-360
            else:
                self.sky_map_tp[i,2]= self.az[i]
            
        
    def init_skycoord(self,coord,t):
        self.obj_a = SkyCoord(coord, unit= (u.hourangle, u.deg))
        self.obj_a_altaz = altaz_sources(self.obj_a,t)
                              
       
    def plot_skymap(self):
        plt.plot(self.sky_map_tp[:,2], self.sky_map_tp[:,1],'*'),#norm = LogNorm())
        plt.ylim([180,0])
        plt.ylabel('theta')
        plt.xlabel('phi')

    def calculate_ha_dec(self,lst):
        self.H_o = lst*15*np.pi/180 - self.obj_a.ra.rad
        self.d =  self.obj_a.dec.rad
        
    def calculate_delphi(self,off,wav,L,M,co_planar=True,w_effect=False):
        if w_effect==True:
            H_o = self.H_o
            d = self.d
        else:
            H_o = 0
            d =0
            
        if co_planar == True:
            x_array = 0
        del_u = np.sin(H_o)*x_array+ np.cos(H_o) *off/wav
        del_v = -np.sin(d)*np.cos(H_o)*x_array + np.cos(d)*off/wav + np.sin(d)*np.sin(H_o)*off/wav

        self.del_phi = 2*np.pi* (del_u*L + del_v*M)
        
    def tp_to_lm(self,theta_lim,l_grid,m_grid):
        self.sky_map=np.zeros((len(l_grid),len(m_grid)))
        for i in range(np.shape(self.sky_map_tp)[0]):
                
            if self.sky_map_tp[i,1]<theta_lim:
                l =np.sin(self.sky_map_tp[i,1]*np.pi/180)*np.cos(self.sky_map_tp[i,2]*np.pi/180)
                m = -np.sin(self.sky_map_tp[i,1]*np.pi/180)*np.sin(self.sky_map_tp[i,2]*np.pi/180)

                l_p = np.where(l_grid>l)[0][0]
                m_p = np.where(m_grid>m)[0][0]

                self.sky_map[m_p,l_p] = self.sky_map_tp[i,0]

     
                
    def make_sky_coherence(self):
        self.sky_coherence = np.array([[self.sky_map, np.zeros_like(self.sky_map)],[np.zeros_like(self.sky_map), self.sky_map]])

    def make_offset_jacob(self,J):
        self.J_b = np.array([[np.ones_like(self.del_phi),np.zeros_like(self.del_phi)] , [np.zeros_like(self.del_phi),np.exp(1j*self.del_phi)]])
        self.J_off = np.zeros_like(self.J_b,dtype=np.complex128)
        
        for a in range(2):
            for b in range(2):

                self.J_off[a,b] = np.sum(self.J_b[a,:]* J[:,b],axis = 0)
                
    def make_pseudo_vis(self,J,w_effect=False):
        vis_temp = np.zeros_like(self.J_off,dtype=np.complex128)
        vis_temp_off = np.zeros_like(self.J_off,dtype=np.complex128)
        
        self.beam_ft = np.zeros((2,2,np.shape(self.del_phi)[0],np.shape(self.del_phi)[1]),dtype=np.complex128)
        self.beam_ft_off = np.zeros((2,2,np.shape(self.del_phi)[0],np.shape(self.del_phi)[1]),dtype=np.complex128)
        
        self.vis = np.zeros_like(self.J_off,dtype=np.complex128)
        self.vis_off = np.zeros_like(self.J_off,dtype=np.complex128)
        roll_factor = int(np.shape(self.del_phi)[0]/2)
        
        for a in range(2):
            for b in range(2):

                vis_temp[a,b]= np.sum(J[a,:] * self.sky_coherence[:,b],axis=0)
                vis_temp_off[a,b]= np.sum(self.J_off[a,:] * self.sky_coherence[:,b],axis=0)
       
        for a in range(2):
            for b in range(2):

                self.vis[a,b]= np.sum(vis_temp[a,:] * np.conj(J[b,:]),axis=0)
                self.vis_off[a,b]= np.sum(vis_temp_off[a,:] * np.conj(self.J_off[b,:]),axis=0)
                if w_effect==True:
                    self.roll_index = np.where(np.abs(self.vis[a,b])==np.max(np.abs(self.vis[a,b])))
                    self.vis[a,b] = np.roll(self.vis[a,b],shift=[roll_factor-self.roll_index[0][0],roll_factor-self.roll_index[1][0]],axis=[0,1]) 
                    self.vis_off[a,b] = np.roll(self.vis_off[a,b],shift=[roll_factor-self.roll_index[0][0],roll_factor-self.roll_index[1][0]],axis=[0,1])
                
                self.beam_ft_off[a,b,:,:] = (fft.fftshift(fft.fft2(fft.fftshift(self.vis_off[a,b]))))
                self.beam_ft[a,b,:,:] = (fft.fftshift(fft.fft2(fft.fftshift(self.vis[a,b]))))
               
                
                
    def uvcal(self,wav,far_array,w_effect=False):
            if w_effect==True:
                H_o = self.H_o
                d = self.d
            else:
                H_o = 0
                d =0
                
            n_basefa = permutations(np.arange(np.shape(far_array)[0]),2)
            basefa = (list(n_basefa))
            for i in range(np.shape(far_array)[0]):
                basefa.append((i,i))

            self.u = np.zeros((len(basefa)))
            self.v = np.zeros((len(basefa)))
            self.w = np.zeros((len(basefa)))

            for i in range(len(basefa)): 
                        yfa = (far_array[basefa[i][1],0] - far_array[basefa[i][0],0])*10**3/wav
                        zfa = (far_array[basefa[i][1],1] - far_array[basefa[i][0],1])*10**3/wav
                        x =0
                        self.u[i] = np.sin(H_o)*x+ np.cos(H_o) * yfa
                        self.v[i] = -np.sin(d)*np.cos(H_o)*x + np.cos(d)*zfa + np.sin(d)*np.sin(H_o)*yfa
                        self.w[i] =  np.cos(d)*np.cos(H_o)*x+ np.sin(d)*zfa - np.cos(d)*np.sin(H_o)*yfa
        
        
    def make_dirty_image(self,w_effect=False,off_c=0):
            factor = int(np.shape(self.del_phi)[0]/2)
            H, xedges, yedges = np.histogram2d((self.u.flatten()),(self.v.flatten()),bins=(factor*2,factor*2),normed = True, range=[[-factor,factor],[-factor,factor]])
            self.image = np.zeros_like(self.beam_ft,dtype = np.complex128) 
            self.image_off = np.zeros_like(self.beam_ft,dtype = np.complex128)
            vis = np.zeros_like(self.beam_ft, dtype = np.complex128)
            vis_off = np.zeros_like(self.beam_ft, dtype = np.complex128)
           
            for j in range(2):
                for i in range(2):
                    vis[i,j] =    H*self.beam_ft[i,j]
                    vis_off[i,j] = H*self.beam_ft_off[i,j]
                
            if off_c ==1:
                vis_offc = offset_correction(self.J_in_ft, vis_off)
            elif off_c==2:
                vis_offc = offset_correction(self.J_in, vis_off)
            else:
                vis_offc = vis_off
                
            for j in range(2):
                for i in range(2):
                    self.image[i,j] = (fft.ifftshift(fft.ifft2(fft.ifftshift(vis[i,j]))))*(factor)**2
                    self.image_off[i,j] = (fft.ifftshift(fft.ifft2(fft.ifftshift(vis_offc[i,j]))))*(factor)**2
                   
                    if w_effect==True:
                        self.image[i,j] = np.roll(self.image[i,j],[-factor+self.roll_index[0][0], -factor+self.roll_index[1][0]],axis=[0,1])
                        self.image_off[i,j] = np.roll(self.image_off[i,j],[-factor+self.roll_index[0][0], -factor+self.roll_index[1][0]],axis=[0,1])
    
    
    def invert_offset_jacob(self):
        self.J_in = np.zeros_like(self.J_b, dtype=np.complex128)
        
        for i  in range(np.shape(self.J_b)[2]):
            for j in range(np.shape(self.J_b)[2]):
                self.J_in[:,:,i,j] = np.linalg.inv(self.J_b[:,:,i,j])
                
    
    def offset_ft(self):
        self.J_in_ft = np.zeros_like(self.J_in,dtype= np.complex128)
        for a in range(2):
            for b in range(2):
                self.J_in_ft[a,b,:,:] = (fft.fftshift(fft.fft2(fft.fftshift(self.J_in[a,b]))))
                
               
        
def offset_correction(J_in, image):
    """
    Corrects for the offset between the X and Y dipoles in the sky images.
    """

    c_image = np.zeros_like(image,dtype = np.complex128)
    c_image_temp = np.zeros_like(J_in,dtype=np.complex128)
    
        
    
    for a in range(2):
        for b in range(2):
            
            c_image_temp[a,b] =  np.sum(J_in[a,:] * image[:,b],axis=0)
            
    for a in range(2):
        for b in range(2):
            c_image[a,b]= np.sum(c_image_temp[a,:] * np.conj(J_in[b,:]),axis=0)
            
    return c_image