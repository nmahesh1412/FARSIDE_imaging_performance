import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from astropy.coordinates import EarthLocation
from astropy import coordinates as coord

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


def muller_cal(etheta_square,ephi_square,etheta_square90,ephi_square90,freq_index,del_u,del_v,offset=0,title='dipole'):
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


    
            
    
    n_M00 = np.max(np.real(M00[:,:]))
    lth=1e-4
    P,T = np.meshgrid(phi,theta)
  

    fig,ax = plt.subplots(4,4, subplot_kw=dict(polar=True),figsize=(20,20))#,constrained_layout=True)
    #fig.suptitle(title,fontsize=30)
    #plt.title('Dipole_100m_regolith_0p1MHz')
    p = ax[0,0].pcolormesh(P,T,np.real(M00[:90,:])/n_M00,cmap='coolwarm',vmax=1,vmin=0)#norm = SymLogNorm(linthresh=lth)#,  norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,0].set_title('M00',{'fontsize':15},pad=15.0)
    ax[0,0].set_yticklabels([])
    plt.colorbar(p,ax=ax[0,0],pad=0.1,shrink=0.8)

    p = ax[0,1].pcolormesh(P,T,np.real(M01[:90,:])/n_M00,cmap='coolwarm',vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh=1e-4, ))#, norm = SymLogNorm(linthresh= lth,vmin = -0.01, vmax = 0.01))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,1].set_title('M01',{'fontsize':15},pad=15.0)
    ax[0,1].set_yticklabels([])
    plt.colorbar(p,ax=ax[0,1],pad=0.1,shrink=0.8)

    p= ax[0,2].pcolormesh(P,T,np.real(M02[:90,:])/n_M00,cmap='coolwarm',vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh=lth, vmin = -1e-3, vmax = 1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,2].set_title('M02',{'fontsize':15},pad=15.0)
    ax[0,2].set_yticklabels([])
    plt.colorbar(p,ax=ax[0,2],pad=0.1,shrink=0.8)

    p = ax[0,3].pcolormesh(P,T,np.real(M03[:90,:])/n_M00,cmap='coolwarm',vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh= lth, vmin = -1e-3, vmax = 1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[0,3].set_title('M03',{'fontsize':15},pad=15.0)
    ax[0,3].set_yticklabels([])
    plt.colorbar(p,ax=ax[0,3],pad=0.1,shrink=0.8)
    ##################################
    p = ax[1,0].pcolormesh(P,T,np.real(M10[:90,:])/n_M00,cmap='coolwarm',vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh= lth, vmin = -1e-3, vmax = 1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[1,0].set_title('M10',{'fontsize':15},pad=15.0)
    ax[1,0].set_yticklabels([])
    plt.colorbar(p,ax=ax[1,0],pad=0.1,shrink=0.8)

    p = ax[1,1].pcolormesh(P,T,np.real(M11[:90,:])/n_M00,cmap='coolwarm',vmin = -1, vmax =1)#norm = SymLogNorm(linthresh=lth, vmin = -1, vmax =1)) #X,Y & data2D must all be same dimensions
    ax[1,1].set_title('M11',{'fontsize':15},pad=15.0)
    ax[1,1].set_yticklabels([])
    plt.colorbar(p,ax=ax[1,1],pad=0.1,shrink=0.8)

    p=ax[1,2].pcolormesh(P,T,np.real(M12[:90,:])/n_M00,cmap='coolwarm',vmin = -1, vmax =1)#norm = SymLogNorm(linthresh=lth, vmin = -1, vmax=1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[1,2].set_title('M12',{'fontsize':15},pad=15.0)
    ax[1,2].set_yticklabels([])
    plt.colorbar(p,ax=ax[1,2],pad=0.1,shrink=0.8)

    p = ax[1,3].pcolormesh(P,T,np.real(M13[:90,:])/n_M00,cmap='coolwarm',vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[1,3].set_title('M13',{'fontsize':15},pad=15.0)
    ax[1,3].set_yticklabels([])
    plt.colorbar(p,ax=ax[1,3],pad=0.1,shrink=0.8)

    ##################################
    p=ax[2,0].pcolormesh(P,T,np.real(M20[:90,:])/n_M00,cmap='coolwarm',vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,0].set_title('M20',{'fontsize':15},pad=15.0)
    ax[2,0].set_yticklabels([])
    plt.colorbar(p,ax=ax[2,0],pad=0.1,shrink=0.8)

    p = ax[2,1].pcolormesh(P,T,np.real(M21[:90,:])/n_M00,cmap='coolwarm',vmin = -1, vmax =1)#norm = SymLogNorm(linthresh=lth, vmin = -1, vmax=1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,1].set_title('M21',{'fontsize':15},pad=15.0)
    ax[2,1].set_yticklabels([])
    plt.colorbar(p,ax=ax[2,1],pad=0.1,shrink=0.8)

    p = ax[2,2].pcolormesh(P,T,np.real(M22[:90,:])/n_M00,cmap='coolwarm',vmin = -1, vmax =1)#norm = SymLogNorm(linthresh=lth, vmin = -1, vmax =1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,2].set_title('M22',{'fontsize':15},pad=15.0)
    ax[2,2].set_yticklabels([])
    plt.colorbar(p,ax=ax[2,2],pad=0.1,shrink=0.8)

    p = ax[2,3].pcolormesh(P,T,np.real(M23[:90,:])/n_M00,cmap='coolwarm',vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[2,3].set_title('M23',{'fontsize':15},pad=15.0)
    ax[2,3].set_yticklabels([])
    plt.colorbar(p,ax=ax[2,3],pad=0.1,shrink=0.8)

    ##################################
    p = ax[3,0].pcolormesh(P,T,np.real(M30[:90,:])/n_M00,cmap='coolwarm',vmin = -1e-3, vmax =1e-3)# norm=SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,0].set_title('M30',{'fontsize':15},pad=15.0)
    ax[3,0].set_yticklabels([])
    plt.colorbar(p,ax=ax[3,0],pad=0.1,shrink=0.8)

    p = ax[3,1].pcolormesh(P,T,np.real(M31[:90,:])/n_M00,cmap='coolwarm',vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,1].set_title('M31',{'fontsize':15},pad=15.0)
    ax[3,1].set_yticklabels([])
    plt.colorbar(p,ax=ax[3,1],pad=0.1,shrink=0.8)

    p = ax[3,2].pcolormesh(P,T,np.real(M32[:90,:])/n_M00,cmap='coolwarm',vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,2].set_title('M32',{'fontsize':15},pad=15.0)
    ax[3,2].set_yticklabels([])
    plt.colorbar(p,ax=ax[3,2],pad=0.1,shrink=0.8)

    p = ax[3,3].pcolormesh(P,T,np.real(M33[:90,:]/n_M00),cmap='coolwarm',vmin = 0, vmax =1)#norm = SymLogNorm(linthresh=lth, vmin = 0, vmax =1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
    ax[3,3].set_title('M33',{'fontsize':15},pad=15.0)
    ax[3,3].set_yticklabels([])
    plt.colorbar(p,ax=ax[3,3],pad=0.1,shrink=0.8)

    plt.savefig('/data4/nmahesh/edges/Lunar/plots/'+title+'.png',dpi=900,bbox_inches = 'tight')




    return M00,M01,M02,M03,M10,M11,M12,M13,M20,M21,M22,M23,M30,M31,M32,M33
    

    def offset_vmuller_cal(freq_index,de,H_o,off,title='dipole'):
  
    phi = np.linspace(0,360,361)*np.pi/180
    theta = np.linspace(0,90,91)*np.pi/180
    wav = np.array([500,150,30])
    freq = 300/wav
    M30 = np.zeros((3,len(theta),len(phi)),dtype='complex')
    M31 = np.zeros((3,len(theta),len(phi)),dtype='complex')
    M32 = np.zeros((3,len(theta),len(phi)),dtype='complex')
    M33 = np.zeros((3,len(theta),len(phi)),dtype='complex')
    lth=1e-4
    
    P,T = np.meshgrid(phi,theta)
    fig,ax = plt.subplots(3,4, subplot_kw=dict(polar=True),figsize=(20,20))#,constrained_layout=True)
   
    for k in range(3):   
        for i in range(len(theta)):
            for j in range(len(phi)):

                l = np.sin(theta[i])*np.cos(phi[j])
                m = np.sin(theta[i]) * np.sin(phi[j])  
                
                del_u,del_v,del_w = ic.uvcal(de,H_o,wav[k],off,off)
                del_phi = 2*np.pi* (del_u*l + del_v*m)
                
                J= np.array([[1,0],[0,np.exp(1j*del_phi)]])



                M30[k,i,j] = np.trace(np.dot(sig_3, J).dot(sig_0).dot(np.conjugate(J.T)))
                M31[k,i,j] = np.trace(np.dot(sig_3, J).dot(sig_1).dot(np.conjugate(J.T)))
                M32[k,i,j] = np.trace(np.dot(sig_3, J).dot(sig_2).dot(np.conjugate(J.T)))
                M33[k,i,j] = np.trace(np.dot(sig_3, J).dot(sig_3).dot(np.conjugate(J.T)))

       

        ##################################
        p = ax[k,0].pcolormesh(P,T,np.real(M30[k,:90,:]),cmap='coolwarm')#,vmin = -1e-3, vmax =1e-3)# norm=SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
        ax[k,0].set_title(str(freq[k]),{'fontsize':15},pad=15.0)
        ax[k,0].set_yticklabels([])
        plt.colorbar(p,ax=ax[k,0],pad=0.1,shrink=0.8)

        p = ax[k,1].pcolormesh(P,T,np.real(M31[k,:90,:]),cmap='coolwarm')#,vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
        #ax[k,1].set_title('M31',{'fontsize':15},pad=15.0)
        ax[k,1].set_yticklabels([])
        plt.colorbar(p,ax=ax[k,1],pad=0.1,shrink=0.8)

        p = ax[k,2].pcolormesh(P,T,np.real(M32[k,:90,:]),cmap='coolwarm')#,vmin = -1e-3, vmax =1e-3)#norm = SymLogNorm(linthresh=lth, vmin = -1e-3, vmax=1e-3))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
        #ax[k,2].set_title('M32',{'fontsize':15},pad=15.0)
        ax[k,2].set_yticklabels([])
        plt.colorbar(p,ax=ax[k,2],pad=0.1,shrink=0.8)

        p = ax[k,3].pcolormesh(P,T,np.real(M33[k,:90,:]),cmap='coolwarm')#,vmin = 0, vmax =1)#norm = SymLogNorm(linthresh=lth, vmin = 0, vmax =1))#,norm = LogNorm()) #X,Y & data2D must all be same dimensions
        #ax[k,3].set_title('M33',{'fontsize':15},pad=15.0)
        ax[k,3].set_yticklabels([])
        plt.colorbar(p,ax=ax[k,3],pad=0.1,shrink=0.8)

        plt.savefig('/data4/nmahesh/edges/Lunar/plots/'+title+'.png',dpi=900,bbox_inches = 'tight')
    
        
        

    #return np.real(M00)/n_M00,np.real(M01)/n_M00,np.real(M02)/n_M00,np.real(M03)/n_M00,np.real(M10)/n_M00,np.real(M11)/n_M00,np.real(M12)/n_M00,np.real(M13)/n_M00,np.real(M20)/n_M00,np.real(M21)/n_M00,np.real(M22)/n_M00,np.real(M23)/n_M00,np.real(M30)/n_M00,np.real(M31)/n_M00,np.real(M32)/n_M00,np.real(M33)/n_M00
    

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