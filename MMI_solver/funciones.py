import pickle
# ahora vamos a generar una funcion para poder propagar los modos 
import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
import gplugins as gp
import gplugins.tidy3d as gt
import shapely as shp
from collections import OrderedDict
from skfem.io.meshio import from_meshio
from femwell.mesh import mesh_from_OrderedDict
import femwell.maxwell.waveguide as fmwg
import gplugins.tidy3d.materials as mat
from tqdm.auto import tqdm
from skfem import (
    ElementDG,
    ElementTriP1,
    ElementVector
)
from tidy3d.constants import C_0
import tidy3d.web as web
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins import waveguide
from tidy3d.plugins.mode.web import run as run_mode_solver
from tidy3d.plugins.dispersion import FastDispersionFitter
import scipy.interpolate

##Funciones Femwell

#definimos la funcion guided_modes, que nos filtra los modos guiados de los calculados por el solver
def guided_modes(modes,clad_material=mat.sio2(1.55), tolerancia=0.01):
    """
    Filter the modes to only include the guided ones.
    """
    guided = []
    for mode in modes:
        if (np.real(mode.n_eff)-clad_material) > tolerancia: #si el neff-ncladd > toerancia, es un modo guiado
            guided.append(mode)
    return guided

def get_TETM(modes):
    """
    Get the TE and TM modes from the list of modes.
    """
    TE = []
    TM = []
    for mode in modes:
        if mode.te_fraction > 0.5: #si la fraccion de TE es mayor que 0.5, es un modo TE
            TE.append(mode)
        elif mode.tm_fraction > 0.5: #si la fraccion de TM es mayor que 0.5, es un modo TM
            TM.append(mode)
    return TE,TM




#en primer lugar definiremos la funcion que nos crea una waveguide, que sera el body del mmi

def create_waveguide(
    core_width =1.0, #anchura del core
    core_thickness=0.5, #altura del core
    cent=0.0, #centro del waveguide
    core_material = mat.sin(1.55),
    clad_material = mat.sio2(1.55),
    wavelength = 1.55,
    num_modes=2,
    slab_width=10.0, #cross section
    slab_thickness=0.1, #cross section
    XY=[-1,-1,1,1] #ventana de simulacion
):
    """
    Create a waveguide object.
    """
    #ahora crearemos la waveguide con poligonos
    #Primero, definimos la region de simulacion
    x0=XY[0]
    x1=XY[2]
    y0=XY[1]
    y1=XY[3]

    window=shp.box(x0, y0, x1, y1) #el area de simulacion

    #ahora definimos la region del core
    core = shp.box(-core_width / 2 + cent, -core_thickness / 2, core_width / 2 + cent, core_thickness / 2) #el area del core, en este caso, al ser un cross section rectangular, es un rectangulo
    oxide = shp.clip_by_rect(window,-np.inf,-np.inf, np.inf, np.inf) #el area del oxide, esta funcion substrae el area de window de todo el espacio y lo rellenara de oxido de silicio

    #ahora definiremos un diccionario con estas regiones
    regions = OrderedDict(
        core=core,
        oxide=oxide,

    )

    #ahora definimos las resoluciones de la malla para ambas regiones
    resolutions = dict(
        core={"resolution":core_thickness/5,"distance" : 0.5},
        oxide={"resolution":0.5,"distance" : 2.0}
    ) 

    #ahora definimos la malla   
    mesh= from_meshio(mesh_from_OrderedDict(regions,resolutions,default_resolution_max=10))

    #ahora definimos el material de la malla
    basis = fmwg.Basis(mesh, fmwg.ElementTriP0()) #cogemos la "basis" de la malla y la definimos como un elemento triangular de orden 0
    eps = basis.zeros() #definimos la epsilon como un array de ceros, para posteriormente definir el material

    for subdomain , n in {
        'core': core_material,
        'oxide': clad_material,
    }.items():
        eps[basis.get_dofs(elements=subdomain)] = n**2 #get_dofs devuelve los indices de los nodos de la malla que pertenecen a la subregion, y le asignamos el material correspondiente

    #por último, una vez definido ya el material, usamos el paquete femwell para simular la guia de onda

    modes = fmwg.compute_modes(basis, eps, wavelength,num_modes=num_modes,order=2) #aqui usamos el paquete femwell para calcular los modos de la guia de onda, le pasamos la malla, el material, la longitud de onda y el numero de modos a calcular
    
    guided_m = guided_modes(modes) #aqui usamos la funcion guided_modes (definida arriba) para filtrar los modos guiados, que son los que nos interesan

    return guided_m, basis


#vamos a generar la funcion que genera un "corte" en 1D de la malla, para ver como queda la guia de onda 
# ya que a la hora de propagar, propagaremos el modo en 1D sobre la distancia, no el modo en 2D (no se como se haria, ni si se puede hacer)

def corte(mode = [],slices=1024, x0 = -4, x1 = 4, polarizacion = 'TE'): #no se si esta bien, pero tiene sentido
    """
    Create a 1D cut of the waveguide.
    """
    #(Ex,Ey),_=mode.basis.interpolate(mode.E) #interpolamos el campo electrico del modo E sobre la malla de la basis
    (et,et_basis),(ez,ez_basis)=mode.basis.split(mode.E) #dividimos el campo electrico entre transversal y longitudinal, tambien sobre la malla de la basis
    new_basis = et_basis.with_element(ElementVector(ElementDG(ElementTriP1()))) #definimos una nueva basis, que sera la que usaremos para el corte
    et_xy=new_basis.project(et_basis.interpolate(et))
    (et_x,et_x_basis),(et_y,et_y_basis)=new_basis.split(et_xy) #dividimos el campo electrico entre x e y, para poder graficar
    #ahora vamos a definir el corte, que sera una serie de puntos en el eje x, y en el eje y, la intensidad del campo electrico
    num_of_slices = slices
    #num_of_slices = 7560

    x0=x0
    x1=x1
    x1_value = 0.0 #coordenada en la que queremos hacer el corte

    slices_points = np.vstack([np.linspace(x0, x1, num_of_slices), x1_value*np.ones(num_of_slices)]) #definimos los puntos en el eje x, y en el eje y

    if polarizacion == 'TE':
        base = et_x_basis.probes(slices_points) #probes devuelve samping points, en los puntos definidos en slices_points, usando la basis para la componente x del campo electrico
        field_sl = base @ et_x #aqui multiplicamos el campo electrico por los puntos de la basis, para obtener el campo electrico en los puntos definidos en slices_points
    elif polarizacion == 'TM':
        base = et_y_basis.probes(slices_points)
        field_sl = base @ et_y
    else:
        field_sl=0
        print('polarization not defined')
    return field_sl, slices_points[0] #devolvemos los puntos de la malla en el eje x (eje y esta fijado) y el campo electrico en esos puntos


#ahora vamos a definir una funcion que nos solapa un modo (el que vendra por la entrada) con los modos del MMI

def overlap(modes,mode_in): #modes son los modos del MMI y mode_in es el modo de entrada 
    num_modos = np.size(modes)
    ovlap = np.zeros((1,num_modos), dtype=complex)
    #definimos un array de ceros, que sera el que contenga los resultados de la superposicion
    #definimos un array de ceros, que sera el que contenga los resultados de la superposicion
    for j, mode in enumerate(modes):
        ovlap[0][j]=mode_in.calculate_overlap(mode) #aqui usamos la funcion calculate_overlap de la clase mode, que nos devuelve el resultado de la superposicion entre el modo de entrada y el modo del MMI
    
    return ovlap

#ahora vamos a definir una funcion que nos propague el modo a lo largo de la longitud del MMI, y nos devuelva el campo electrico en la salida
def propagate(wvl = 1.55, L = 100.0, overlap_in= [],modes_mmi = [], slices = [],n_in=2): #modes son los modos del MMI y mode_in es el modo de entrada 
    """
    Propagate the mode through the MMI.
    """
    num_modos = np.size(modes_mmi)
    #dz = L/1000 #definimos el paso de propagacion, en este caso, 1/1000 de la longitud del MMI
    pasos = 1000
    L_v=np.linspace(0,L,pasos) #definimos la longitud de la propagacion, en pasos

    #ahora definiremos las constantes de propagacion beta y la fase

    beta = np.zeros((num_modos))
    for i, mode in enumerate(modes_mmi):
        beta[i] = np.real(mode.n_eff) * 2 * np.pi / wvl #aqui usamos la funcion neff de la clase mode, que nos devuelve el neff del modo, y lo multiplicamos por 2*pi/lambda para obtener la constante de propagacion
    
    phase = np.zeros((num_modos),dtype=complex) #definimos un array de ceros, que sera el que contenga la fase
    propagacion = np.zeros((num_modos),dtype=complex) #definimos un array de ceros, que sera el que contenga la propagacion
    num_puntos= np.size(slices,1) #definimos el numero de puntos en el eje x
    intensidad_L = np.zeros((pasos,n_in,num_puntos),dtype=complex) #definimos un array de ceros, que sera el que contenga la intensidad en funcion de la longitud de propagacion y en funcion del punto en el eje x
    #ya que ez "esta ubicado" del eje x 
    for i , L in enumerate(tqdm(L_v)):
        
        phase = np.exp(1j*beta*L)#aqui definimos la fase como la exponencial de la constante de propagacion por la longitud de propagacion
        
        propagacion = np.multiply(overlap_in,phase.T) #aqui multiplicamos la superposicion por la fase, para obtener la propagacion
        
        ez = np.zeros((n_in,num_puntos), dtype=complex) #definimos un array de ceros, que sera el que contenga el campo electrico
        for j,mo in enumerate(modes_mmi):
            ez += slices[j] * propagacion[j]
        intensidad_L[i] = np.abs(ez)**2 #aqui definimos la intensidad como el cuadrado del campo electrico
    return L_v ,intensidad_L, propagacion   #devolvemos la intensidad y la longitud de propagacion, y propagacion, que es el campo a la salida de la propagacion
     
     #el tqdm es para la barrita de progreso

##Funciones MMI Femwell

#ahora vamos a definir la funcion que nos crea el mmi, pero antes, pondremos las funciones que vayamos a utilizar dentro de la funcion del MMI
def mode_finder_MMI(MMI_width=10.0, wg_array_thickness=1.0,  wvl=1.55, mat_core=mat.sin(1.55), mat_clad=mat.sio2(1.55), MMI_num_modes=10):
    x0 = -4 - MMI_width / 2
    x1 = 4 + MMI_width / 2
    y0 = -2
    y1 = 2
    XY = [x0, y0, x1, y1]
    #resolvemos con la funcion waveguide anteriormente hecha, y con los parametros del mmi
    MMI_modes, MMI_basis = create_waveguide(
        core_width=MMI_width,
        core_thickness=wg_array_thickness,
        core_material=mat_core,
        clad_material=mat_clad,
        wavelength=wvl,
        num_modes=MMI_num_modes,
        slab_width=0.0,
        slab_thickness=0.0,
        XY=XY
    )

    return MMI_modes, MMI_basis ,XY



def L_pi(MMI_modes, wvl): #ademas de ser un getTETM
    neff_te = []
    neff_tm = []
    MMI_modes_TE = []
    MMI_modes_TM = []
    Lpi_TE = 0
    Lpi_TM = 0
    for i, mode in enumerate(MMI_modes):
        if mode.te_fraction > 0.5:
            neff_te.append(mode.n_eff) 
            MMI_modes_TE.append(mode)
        if mode.tm_fraction > 0.5:
            neff_tm.append(mode.n_eff)  
            MMI_modes_TM.append(mode)

    Lpi_TE=wvl * 0.5 / np.real(neff_te[0]-neff_te[1])
    Lpi_TM=wvl * 0.5 / np.real(neff_tm[0]-neff_tm[1])

    return Lpi_TE, Lpi_TM,neff_te, neff_tm, MMI_modes_TE, MMI_modes_TM

def cortes_todos(MMI_modes_TE=[], MMI_modes_TM=[],x0=-4,x1=4): #coge cada modo y lo "slicea" en 1024 partes, OBLIGATORIO PASARLE x0 Y x1
    MMI_sliced_TE = []
    MMI_sliced_TM = []
    punts=[] #inicializo porque sino me da una especie de error
    if np.size(MMI_modes_TE)>1: 
        for i, mode in enumerate(MMI_modes_TE):
            a , b = corte(mode, slices=1024, x0=x0, x1=x1, polarizacion='TE')
            MMI_sliced_TE.append(a)
            if i == 0:
                punts=b
    else:
        a , b = corte(MMI_modes_TE, slices=1024, x0=x0, x1=x1, polarizacion='TE')
        MMI_sliced_TE.append(a)
        punts=b
    if np.size(MMI_modes_TM)>1: 
        for i, mode in enumerate(MMI_modes_TM):
            a , b = corte(mode, slices=1024, x0=x0, x1=x1, polarizacion='TM')
            MMI_sliced_TM.append(a)
            if i == 0:
                punts=b
    else:
        a , b = corte(MMI_modes_TM, slices=1024, x0=x0, x1=x1, polarizacion='TE')
        MMI_sliced_TM.append(a)
        punts=b
    
    # MMI_sliced_TE[0][:] deberia ser el primer modo TE "sliceado" en 1024 partes sobre el eje x (1024 funciones sobre el eje y)
    
    return MMI_sliced_TE, MMI_sliced_TM , punts

#ahora vamos a definir la funcion que nos solapa los modos de entrada y salida con los modos del MMI
'''
def overlap_mmi(modes_mmi, mode_in, mode_out): #los modos del MMI que se pasan los los sliceados

    overlap_in = np.zeros((2, np.size(modes_mmi)), dtype=complex) #definimos un array de ceros, que sera el que contenga los resultados de la superposicion. El dos es porque entramos con 2 waveguides (y salimos), y luego el numero de modos del waveguide. Aunque generalmente usaremos el primero
    overlap_out = np.zeros((2, np.size(modes_mmi)), dtype=complex) #definimos un array de ceros, que sera el que contenga los resultados de la superposicion. El dos es porque entramos con 2 waveguides (y salimos), y luego el numero de modos del waveguide. Aunque generalmente usaremos el primero

    #ahora vamos a hacer el solapamiento entre los modos de entrada y los modos del MMI
    
    overlap_in = overlap(modes_mmi, mode_in) #aqui usamos la funcion overlap (definida arriba) para filtrar los modos guiados, que son los que nos interesan
    #lo mismo con la salida
    overlap_out = overlap(modes_mmi, mode_out) 
    
    return overlap_in, overlap_out #creo que no la uso
'''
#ahora que tenemos la funcion de solapamiento, vamos a definir la funcion que nos calcule la propagacion del modo a lo largo de la longitud del MMI
def propagacion_mmi(wvl,L,overlap_in,modes_mmi,slices): #los modos del MMI que se pasan los los sliceados
    
    L_v, intensidad_L , campo_salida = propagate(
        wvl, 
        L, 
        overlap_in, #a este le tengo que pasar el overlap_in que sale de la funcion overlap_mmi 
        modes_mmi, 
        slices) #aqui usamos la funcion propagate (definida arriba) para filtrar los modos guiados, que son los que nos interesan
    return L_v, intensidad_L, campo_salida


#ahora tenemos que definir la funcion que transfiere el campo propagado en el MMI a la salida

def output_field(overlap_out=[], campo_salida_mmi=[],overlap_in=[]): #los modos del MMI que se pasan los los sliceados
    power_out = np.zeros(2, dtype=complex) #definimos un array de ceros, que sera el que contenga los resultados de la superposicion. El dos es porque entramos con 2 waveguides (y salimos), 
    phase_out = np.zeros(2, dtype=complex) #definimos un array de ceros, que sera el que contenga los resultados de la superposicion. El dos es porque entramos con 2 waveguides (y salimos), 
    field_out = np.zeros(2, dtype=complex) #definimos un array de ceros, que sera el que contenga los resultados de la superposicion. El dos es porque entramos con 2 waveguides (y salimos),
    #ahora vamos a hacer el solapamiento entre los modos de salida y los modos del MMI
    for i, ovl_out in enumerate(overlap_out):
        
        field = campo_salida_mmi @ ovl_out #multiplicamos el campo a la salida del mmi por la integral de solapamiento entre modos. Al ser multimodo el MMI, esto se hace con la multiplicacion de matrices
#        field_out[i] = field #aqui guardamos el campo a la salida del mmi
        power_out[i] = np.abs(field)**2
   #     phase_out[i] = np.angle(field) #aqui guardamos la fase a la salida del mmi, esta en radianes
    #ahora tenemos que normalizar la potencia de entrada, para que sume 1, y poder ver que fraccion de potencia sale
    total_power_in = np.sum(np.abs(overlap_in[0])**2) #aqui guardamos la potencia total de entrada del primer modo unicamente(suponemos que la entrada es single-mode), que es la suma de las potencias de los modos de entrada
    total_power_out = np.sum(power_out) #aqui guardamos la potencia total de salida del primer modo unicamente(suponemos que la salida es single-mode), que es la suma de las potencias de los modos de salida
    
    excess_loss = 10 * np.log10(1 / total_power_out) #aqui guardamos la perdida de potencia, que es la diferencia entre la potencia de entrada y la potencia de salida, en dB
    
    ratio_out = power_out / total_power_out #aqui guardamos la potencia de salida de cada waveguide de salida, que es la potencia de salida de cada wvg dividido por la potencia total de salida
    return excess_loss, ratio_out #devolvemos la potencia de salida de cada waveguide de salida, que es la potencia de salida de cada wvg dividido por la potencia total de salida

def io_waveguide_mode_finder(XY = [], wg_width=10.0, wg_array_thickness=1.0,  wvl=1.55, mat_core=mat.sin(1.55), mat_clad=mat.sio2(1.55), MMI_num_modes=10):
    in_wvg_modes, in_wvg_basis = create_waveguide(
        core_width=wg_width,
        core_thickness=wg_array_thickness,
        core_material=mat_core,
        clad_material=mat_clad,
        wavelength=wvl,
        num_modes=MMI_num_modes,
        slab_width=0.0,
        slab_thickness=0.0,
        XY=XY
    )
    in_TE,in_TM = get_TETM(in_wvg_modes)
    
    if len(in_TE) > 0:
        TE_mode = in_TE[0]
    if len(in_TM) > 0:  
        TM_mode = in_TM[0]
    return in_TE,in_TM

def io_waveguide_mode_sliced(XY=[],wg_width=10.0, wg_array_thickness=1.0,  wvl=1.55, mat_core=mat.sin(1.55), mat_clad=mat.sio2(1.55), MMI_num_modes=10):
    in_wvg_modes, in_wvg_basis = create_waveguide(
        core_width=wg_width,
        core_thickness=wg_array_thickness,
        core_material=mat_core,
        clad_material=mat_clad,
        wavelength=wvl,
        num_modes=MMI_num_modes,
        slab_width=0.0,
        slab_thickness=0.0,
        XY=XY
    )
    
    in_TE,in_TM = get_TETM(in_wvg_modes)

    x0=XY[0]
    x1=XY[2]
    TE_mode_sliced = []
    TM_mode_sliced = []
    x=[]
    if len(in_TE) > 0:
        TE_mode = in_TE[0]
        TE_mode_sliced,x= corte(TE_mode,x0=x0, x1=x1)
    if len(in_TM) > 0:  
        TM_mode = in_TM[0]
        TM_mode_sliced,x= corte(TM_mode, x0=x0, x1=x1)

    return TE_mode_sliced,TM_mode_sliced,x


#ahora vamos a definir una funcion que nos shiftee estos modos a la entrada / salida del MMI


def shift_modes(MMI_wd=10.0, slices = 1024,n_in=2,n_out=2, TE_mode_sliced=[], TM_mode_sliced=[],x=[], input_positions=[], output_positions=[]):

    all_in_wvg_sliced_TE = np.zeros((n_in,slices),dtype=complex)
    all_out_wvg_sliced_TE = np.zeros((n_in,slices),dtype=complex)

    all_in_wvg_sliced_TM = np.zeros((n_in,slices),dtype=complex)
    all_out_wvg_sliced_TM = np.zeros((n_in,slices),dtype=complex)
    #mismo procedimiento para todos
    for i in range(n_in):
        shift = input_positions[i]*MMI_wd # cojo la posicion del input wvg
        shift_pts = int(shift / (x[1]-x[0])) 
        shifted_mode_TE = np.roll(TE_mode_sliced, shift_pts) #shifteo el modo tantos puntos de la malla como este la wvg
        all_in_wvg_sliced_TE[i] = shifted_mode_TE  #aqui guardamos el modo shifteado en la posicion de la wvg de entrada, multiplicado por la distribucion de potencia
    for i in range(n_out):

        shift = output_positions[i]*MMI_wd

        shift_pts = int(shift / (x[1]-x[0])) 
        shifted_mode_TE = np.roll(TE_mode_sliced, shift_pts)
        all_out_wvg_sliced_TE[i] = shifted_mode_TE
    
    for i in range(n_in):
        shift = input_positions[i]*MMI_wd
        shift_pts = int(shift / (x[1]-x[0])) 
        shifted_mode_TM = np.roll(TM_mode_sliced,shift_pts)
        all_in_wvg_sliced_TM[i] = shifted_mode_TM 
    for i in range(n_out):
        shift = output_positions[i]*MMI_wd
        shift_pts = int(shift / (x[1]-x[0])) 
        shifted_mode_TM = np.roll(TM_mode_sliced, shift_pts)
        all_out_wvg_sliced_TM[i] = shifted_mode_TM

    return all_in_wvg_sliced_TE, all_out_wvg_sliced_TE, all_in_wvg_sliced_TM, all_out_wvg_sliced_TM

def normalizar_slice_wvg(modos=[], x=[]):
    for i, modo in enumerate(modos):
        integral = np.trapezoid(np.abs(modo)**2, x) #aqui usamos la funcion trapezoid de numpy para calcular la integral del modo, que es la potencia total del modo, de forma numerica
        modos[i] = modo / np.sqrt(integral) #aqui normalizamos el modo, dividiendo el modo por la raiz de la integral
    return modos

def normalizar_slices_mmi(modos_mmi=[], x=[]):
    for i, modo in enumerate(modos_mmi):
        integral = np.trapezoid(np.abs(modo)**2, x)
        modos_mmi[i] = modo / np.sqrt(integral)
    return modos_mmi
def overlap_sliced_modes(modo1=[], modo2=[], x=[]):
    overlap = np.trapezoid(np.conjugate(modo1) * modo2, x)
    return overlap

def overlap_sliced_IO(modo_in_norm=[], modo_out_norm=[], modos_mmi_norm=[], x=[]):
    n_IN = np.size(modo_in_norm,0)
    n_OUT = np.size(modo_out_norm,0)
    n_Modes = np.size(modos_mmi_norm,0)
    overlap_in = np.zeros((n_IN, n_Modes), dtype=complex) #definimos un array de ceros, que sera el que contenga los resultados de la superposicion. El dos es porque entramos con 2 waveguides (y salimos), y luego el numero de modos del waveguide. Aunque generalmente usaremos el primero
    
    for i,s1 in enumerate(modo_in_norm):
        for j,s2 in enumerate(modos_mmi_norm):
            overlap_in[i][j] = overlap_sliced_modes(s1,s2,x)
    
    overlap_out = np.zeros((n_OUT, n_Modes), dtype=complex)
    for i,s1 in enumerate(modo_out_norm):
        for j,s2 in enumerate(modos_mmi_norm):
            overlap_out[i][j] = overlap_sliced_modes(s1,s2,x)

    return overlap_in, overlap_out


def MMI(
    L_MMI = 100.0, #longitud del MMI
    wvl = 1.55, #longitud de onda
    mat_core = mat.sin(1.55), #material del core
    mat_clad = mat.sio2(1.55), #material del clad
    input_number = 2, #número de waveguides de entrada
    output_number = 2, #número de waveguides de salida
    input_positions = [-1/6,1/6], #posiciones de las waveguides de entrada
    output_positions = [-1/6,1/6],#posiciones de las waveguides de salida#la maxima posicion es [-1/2,1/2]
    wg_array_width=1.0, #anchura de la waveguide de entrada/salida
    wg_array_thickness=0.5, #altura de la waveguide de entrada/salida, es igual que la del MMI
    MMI_width=6.0, #anchura del MMI
    MMI_num_modes=20, #número de modos del MMI a calcular
    slices=1024,
    gap = 0.8 #distancia entre waveguides de entrada/salida
):
    #esta hecho para que la salida de la funcion sea para TE, pero se puede ampliar para hacer TM, solo habria que sustituir TE por TM
    #calculo de modos
    MMI_modes , MMI_basis , XY = mode_finder_MMI(MMI_width, wg_array_thickness,  wvl, mat_core, mat_clad, MMI_num_modes)
    MMI_modes_TE, MMI_modes_TM = get_TETM(MMI_modes)
    Lpi_TE, Lpi_TM,_, _, _, _=L_pi(MMI_modes, wvl)
    
    #Lpi_TE, Lpi_TM,neff_te, neff_tm, MMI_modes_TE, MMI_modes_TM = L_pi(MMI_modes, wvl)

    TE_mode_in, TM_mode_in=io_waveguide_mode_finder(XY,wg_array_width, wg_array_thickness,  wvl, mat_core, mat_clad, MMI_num_modes)
    TE_mode_out,TM_mode_out= io_waveguide_mode_finder(XY,wg_array_width, wg_array_thickness,  wvl, mat_core, mat_clad, MMI_num_modes)
    
    MMI_modes_TE_sl, MMI_modes_TM_sl, _ = cortes_todos(MMI_modes_TE, MMI_modes_TM,x0=XY[0],x1=XY[2]) #aqui usamos la funcion cortes_todos (definida arriba) para filtrar los modos guiados, que son los que nos interesan
    TE_mode_in_sl, TM_mode_in_sl,x= io_waveguide_mode_sliced(XY,wg_array_width, wg_array_thickness,  wvl, mat_core, mat_clad, MMI_num_modes)
    TE_mode_out_sl, TM_mode_out_sl,x= io_waveguide_mode_sliced(XY,wg_array_width, wg_array_thickness,  wvl, mat_core, mat_clad, MMI_num_modes)
    all_in_wvg_sliced_TE, _, all_in_wvg_sliced_TM, _ = shift_modes(MMI_width, slices ,input_number,output_number, TE_mode_in_sl, TM_mode_in_sl,x, input_positions, output_positions)
    _, all_out_wvg_sliced_TE, _, all_out_wvg_sliced_TM = shift_modes(MMI_width, slices ,input_number,output_number, TE_mode_out_sl, TM_mode_out_sl,x, input_positions, output_positions)
    plt.plot(x, np.abs(TE_mode_in_sl)**2)
    plt.plot(x, np.abs(all_in_wvg_sliced_TE[0])**2)
    plt.plot(x, np.abs(all_in_wvg_sliced_TE[1])**2)
    plt.plot(x, np.abs(all_out_wvg_sliced_TE[0])**2)
    plt.plot(x, np.abs(all_out_wvg_sliced_TE[1])**2)
    
    plt.axvline(0)

    print(x)

    #ahora normalizamos los modos de entrada y salida
    all_in_wvg_sliced_TE = normalizar_slice_wvg(all_in_wvg_sliced_TE, x)
    all_out_wvg_sliced_TE = normalizar_slice_wvg(all_out_wvg_sliced_TE, x)
    all_in_wvg_sliced_TM = normalizar_slice_wvg(all_in_wvg_sliced_TM, x)
    all_out_wvg_sliced_TM = normalizar_slice_wvg(all_out_wvg_sliced_TM, x)
    #ahora normalizamos los modos del MMI

    MMI_modes_TE_sl = normalizar_slices_mmi(MMI_modes_TE_sl, x)
    MMI_modes_TM_sl = normalizar_slices_mmi(MMI_modes_TM_sl, x)
    #calculamos los overlaps
    overlap_in_TE, overlap_out_TE = overlap_sliced_IO(all_in_wvg_sliced_TE, all_out_wvg_sliced_TE, MMI_modes_TE_sl, x )
    overlap_in_TM, overlap_out_TM = overlap_sliced_IO(all_in_wvg_sliced_TM, all_out_wvg_sliced_TM, MMI_modes_TM_sl, x )
    #TE_mode_in_sliced, TM_mode_in_sliced, punts = cortes_todos(TE_mode_in, TM_mode_in,x0=XY[0],x1=XY[2]) #aqui usamos la funcion cortes_todos (definida arriba) para filtrar los modos guiados, que son los que nos interesan
    #TE_mode_out_sliced, TM_mode_out_sliced, _ = cortes_todos(TE_mode_out, TM_mode_out,x0=XY[0],x1=XY[2]) #aqui usamos la funcion cortes_todos (definida arriba) para filtrar los modos guiados, que son los que nos interesan
    #overlap_in, overlap_out = overlap_mmi(MMI_modes_TE,TE_mode_in,TE_mode_out) #aqui usamos la funcion overlap_mmi (definida arriba) para filtrar los modos guiados, que son los que nos interesan
    #propagacion y salida de modos
    L_v, intensidad_L, campo_salida_mmi = propagacion_mmi(wvl,L_MMI,overlap_in_TE[1],MMI_modes_TE,MMI_modes_TE_sl) # con este [0] defino cual es el modo que entra en la wvg
    #L_v, intensidad_L, campo_salida_mmi = propagacion_mmi(wvl,L_MMI,overlap_in_TE,MMI_modes_TE,MMI_modes_TE_sl) # con este [0] defino cual es el modo que entra en la wvg
    excess_loss, ratio_out = output_field(overlap_out_TE, campo_salida_mmi,overlap_in_TE)
    if gap<0:
        print('error: gap negativo')
    return ratio_out ,excess_loss, L_v, intensidad_L, Lpi_TE, Lpi_TM

        
##Funciones tidy3d

#definimos el waveguide y los tapers
def create_ridge(w0,x0,y0,y1,z0,z1,angle):
    width = w0
    ridge=td.PolySlab(
        vertices = [(x0-width/2,y0),(x0+width/2,y0),(x0+width/2,y1),(x0-width/2,y1)],
        axis = 2,
        reference_plane="top",
        sidewall_angle = angle,
        slab_bounds = ([z0,z1]),
    )
    box = td.Box(center=(x0,y0 + (y1 - y0)/2,z0 + (z1-z0)/2), size=(np.inf, y1-y0, z1-z0))

    return ridge * box

def create_ridge_butterfly_ACD(w0,x0,y0,y1,z0,z1,angle,central_width_variation):
    width = w0
    ridge=td.PolySlab(
        vertices = [(x0-width/2,y0),(x0+width/2,y0),(x0+width/2-central_width_variation/2,0),(x0+width/2,y1),(x0-width/2,y1),(x0-width/2+central_width_variation/2,0)],
        axis = 2,
        reference_plane="top",
        sidewall_angle = angle,
        slab_bounds = ([z0,z1]),
    )
    box = td.Box(center=(x0,y0 + (y1 - y0)/2,z0 + (z1-z0)/2), size=(np.inf, y1-y0, z1-z0))

    return ridge * box

def create_ridge_butterfly_B(w0,x0,y0,y1,z0,z1,angle,central_width_variation):
    width = w0
    ridge=td.PolySlab(
        vertices = [(x0-width/2,y0),(x0+width/2,y0),(x0+width/2-central_width_variation/2,y0/2),(x0+width/2,0),(x0+width/2-central_width_variation/2,y1/2),(x0+width/2,y1),(x0-width/2,y1),(x0-width/2+central_width_variation/2,y1/2),(x0-width/2,0),(x0-width/2+central_width_variation/2,y0/2)],
        axis = 2,
        reference_plane="top",
        sidewall_angle = angle,
        slab_bounds = ([z0,z1]),
    )
    box = td.Box(center=(x0,y0 + (y1 - y0)/2,z0 + (z1-z0)/2), size=(np.inf, y1-y0, z1-z0))

    return ridge * box

def create_ridge2(w0,x0,y0,y1,z0,z1,angle,len_corner):
    width = w0
    width_corner = width / 6
    
    ridge=td.PolySlab(
        vertices = [(x0-width/2,y0+len_corner),(x0-width/2+width_corner,y0),(x0+width/2-width_corner,y0),(x0+width/2,y0+len_corner),(x0+width/2,y1-len_corner),(x0+width/2-width_corner,y1),(x0-width/2+width_corner,y1),(x0-width/2,y1-len_corner)],
        axis = 2,
        reference_plane="top",
        sidewall_angle = angle,
        slab_bounds = ([z0,z1]),
    )
    box = td.Box(center=(x0,y0 + (y1 - y0)/2,z0 + (z1-z0)/2), size=(np.inf, y1-y0, z1-z0))

    return ridge * box


def create_taper(w0,x0,y0,y1,z0,z1,angle,w_thin):
    width   = w0
    taper = td.PolySlab(
        vertices = [(x0-w_thin/2,y0),(x0+w_thin/2,y0),(x0+width/2,y1),(x0-width/2,y1)],
        axis = 2,
        reference_plane="top",
        sidewall_angle = angle,
        slab_bounds = ([z0,z1]),
    )
    return taper

#definimos las funciones para hacer una simulacion 2.5D


def var_eps_eff(point, ref_point,sim,wavelength=1.55,inf=1000,min_n=1):

    freq = td.C_0 / wavelength

    sim_2d_center = (ref_point[0],ref_point[1],0,)

    sim_2d_size = (0,inf,inf,)

    sim_2d = sim.updated_copy(
        center = sim_2d_center,
        size = sim_2d_size,
        sources = [],
        monitors = [],
        symmetry = (0,0,0),
        boundary_spec = sim.boundary_spec.updated_copy(x=td.Boundary.periodic()),

    )

    #ahora resolveremos el modo en el punto de referencia

    mode_solver_plane = td.Box( center = sim_2d.center , size = (td.inf , 0 , td.inf))

    mode_solver = td.plugins.mode.ModeSolver(
        simulation = sim_2d,
        plane = mode_solver_plane,
        mode_spec = td.ModeSpec(num_modes=1),
        freqs = [freq],
    )

    #como queremos una buena precision, usaremos el modo remoto
    mode_data_ref = run_mode_solver(mode_solver)
    
    #ahora obtenemos el n_eff del mode solver
    neff = mode_data_ref.n_eff.item()

    if point == ref_point:
        return neff**2
    
    #ahora tenemos que calcular el n_eff en el punto de referencia
    x,y = ref_point
    eps_ref = sim.epsilon(
        box = td.Box(center = (x,y,list(sim.center)[2]), size = (0,0,td.inf)), freq = freq
    )

    x,y = point
    eps = sim.epsilon(
        box = td.Box(center = (x,y,list(sim.center)[2]), size = (0,0,td.inf)), freq = freq
    )

    eps_dif = np.squeeze(eps.values) - np.squeeze(eps_ref.values) #aqui calculamos la diferencia de epsilon entre el punto de referencia y el punto donde queremos calcular el n_eff

    z_coord = eps_ref.z.values
    mode_profile = mode_data_ref.Ex 
    Mz2 = scipy.interpolate.interp1d(
        x = mode_profile.z.values,
        y = np.abs(np.squeeze(mode_profile.values))**2
    )
    m_values = Mz2(z_coord)

    num,denom = np.trapezoid(y = eps_dif * m_values, x = z_coord), np.trapezoid(y = m_values, x = z_coord)
    
    if neff**2 +num/denom < min_n:
        return min_n 
    return neff**2 +num/denom

def approximate_material(sim_3D,approx_point,ref_point,spectrum,min_n=1):

    eps = []

    for wl in spectrum:
        eps.append(var_eps_eff(approx_point, ref_point, sim_3D, wavelength=wl, min_n=min_n))

    fitter = FastDispersionFitter(wvl_um = spectrum, n_data = np.sqrt(np.real(eps)))

    medium , rms_error= fitter.fit()

    fig,ax = plt.subplots(1,1,figsize=(3,3))
    fitter.plot(medium,ax=ax)
    ax.set_title("Medium")
    plt.show()
    return medium

def create_2D_sim(sim_3D,new_mediums):

    new_structures = []
    for structure in sim_3D.structures:
        new_structures.append(structure.updated_copy(medium = new_mediums[0]))

    new_size = list(sim_3D.size)
    new_size[2] = 0
    
    new_symmetry = list(sim_3D.symmetry)
    if new_symmetry != [0,0,0]:
        print("Warning: 2D simulation with symmetry, this may not work as expected")
    
    new_symmetry = [0,0,0] #forzamos a que sea 2D

    sim_2D = sim_3D.updated_copy(
        size = new_size,
        symmetry = new_symmetry,
        structures = new_structures,
        medium = new_mediums[1],
        boundary_spec = sim_3D.boundary_spec.updated_copy(z=td.Boundary.periodic()),
    )

    return sim_2D

#definimos la simulacion
def create_2D_MMI_simulation(wvlenth,Len_MMI,MMI_width, wg_array_thickness, wg_array_width,wvg_length, gap, taper_length, freq0, fwidth, sin, sio2,freqs,len_corner):
    MMI_body = td.Structure(
    geometry = create_ridge2(MMI_width,0,-Len_MMI/2,Len_MMI/2,-wg_array_thickness/2,wg_array_thickness/2,0,len_corner),
    medium = sin,)

    Wg_in0 = td.Structure(
        geometry = create_ridge(wg_array_width,-gap/2,-(Len_MMI/2+taper_length+wvg_length),-(Len_MMI/2+taper_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_in0 = td.Structure(
        geometry = create_taper(MMI_width/3,-gap/2,-(Len_MMI/2+taper_length),-(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    Wg_in1 = td.Structure(
        geometry = create_ridge(wg_array_width,gap/2,-(Len_MMI/2+taper_length+wvg_length),-(Len_MMI/2+taper_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_in1 = td.Structure(
        geometry = create_taper(MMI_width/3,gap/2,-(Len_MMI/2+taper_length),-(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    Wg_out0 = td.Structure(
        geometry = create_ridge(wg_array_width,-gap/2,(Len_MMI/2+taper_length),(Len_MMI/2+taper_length+wvg_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_out0 = td.Structure(
        geometry = create_taper(MMI_width/3,-gap/2,(Len_MMI/2+taper_length),(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    Wg_out1 = td.Structure(
        geometry = create_ridge(wg_array_width,gap/2,(Len_MMI/2+taper_length),(Len_MMI/2+taper_length+wvg_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_out1 = td.Structure(
        geometry = create_taper(MMI_width/3,gap/2,(Len_MMI/2+taper_length),(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    ####

    #definimos los monitores y fuentes

    mode_spec = td.ModeSpec(
        num_modes=2,
        target_neff=3,
        track_freq="central",
        precision= "double",
        group_index_step=True
    )
    mode_source = td.ModeSource(
        center = (-gap/2,-(Len_MMI/2+taper_length+1.5),0),
        size = (3 * wg_array_width,0 , 5*wg_array_thickness),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction = "+",
        mode_spec = mode_spec,
        mode_index=0,
        num_freqs=5,
    )
    field_monitor1 = td.FieldMonitor(
        center = (0,0,0), size = (td.inf,td.inf,0), freqs=[freq0], name = "field1"
    )
    field_monitor11 = td.FieldMonitor(
        center = (0,0,wg_array_thickness/2), size = (td.inf,td.inf,0), freqs=[freq0], name = "field11"
    )
    field_monitor2 = td.FieldMonitor(
        center = (0,0,0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field2"
    )
    field_monitor3 = td.FieldMonitor(
        center = (0,Len_MMI/4,0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field3"
    )
    field_monitor4 = td.FieldMonitor(
        center = (0,-Len_MMI/4,0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field4"
    )

    field_monitor5 = td.FieldMonitor(
        center = (0,(Len_MMI/2+taper_length+1.5),0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field5"
    )

    flux_monitor0 = td.FluxMonitor(
        center = (-gap/2,-(Len_MMI/2+taper_length+1.4),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux0",
    )


    flux_monitor1 = td.FluxMonitor(
        center = (-gap/2,(Len_MMI/2+taper_length+1.5),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux1",
    )

    mode_monitor1 = td.ModeMonitor(
        center = (-gap/2, (Len_MMI/2+taper_length+3/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode1",

    )

    flux_monitor2 = td.FluxMonitor(
        center = (gap/2,(Len_MMI/2+taper_length+1.5),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux2",
    )

    mode_monitor2 = td.ModeMonitor( 
        center = (gap/2, (Len_MMI/2+taper_length+3/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode2",
    )
    flux_monitor00 = td.FluxMonitor(
        center = (-gap/2,-(Len_MMI/2),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux00",
    )

    mode_monitor00 = td.ModeMonitor(
        center = (-gap/2, -(Len_MMI/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode00",

    )

    flux_monitor11 = td.FluxMonitor(
        center = (-gap/2,(Len_MMI/2),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux11",
    )

    mode_monitor11 = td.ModeMonitor(
        center = (-gap/2, (Len_MMI/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode11",

    )

    flux_monitor22 = td.FluxMonitor(
        center = (gap/2,(Len_MMI/2),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux22",
    )

    mode_monitor22 = td.ModeMonitor( 
        center = (gap/2, (Len_MMI/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode22",
    )

    Lx = 1.5*MMI_width 
    Ly= 1.5*(Len_MMI+taper_length) 
    Lz = 2*wg_array_thickness
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=15,wavelength=1.55)
    sim = td.Simulation(
        size = (Lx,Ly,Lz),
        grid_spec = grid_spec,
        run_time = 3e-12,
        boundary_spec= td.BoundarySpec.all_sides(boundary = td.PML()),
        medium=sio2,
        structures=(MMI_body,Wg_in0,Taper_in0,Wg_in1,Taper_in1,Wg_out0,Taper_out0,Wg_out1,Taper_out1),
        sources=[mode_source],
        monitors = [field_monitor1,field_monitor2,field_monitor3,field_monitor4,field_monitor5,flux_monitor0,flux_monitor1,flux_monitor2]#,mode_monitor1,mode_monitor2,flux_monitor00,flux_monitor11,flux_monitor22,mode_monitor00,mode_monitor11,mode_monitor22],
    )

    reference_point = (0,0)
    waveguide_point = (0,0)
    other_point = (0,(Len_MMI/2+taper_length+1.5))

    spectrum = wvlenth[::10]

    waveguide_medium = approximate_material(sim, waveguide_point, reference_point, spectrum)
    background_medium = approximate_material(sim, other_point, reference_point, spectrum)

    sim_2D = create_2D_sim(sim, [waveguide_medium, background_medium])
    sim_2D.plot_eps(z=0,freq=freq0)

    job = web.Job(simulation=sim_2D, task_name="MMI_2D", verbose=True)
    sim_data = job.run(path="data/MMI_2D.hdf5")

    T1 = sim_data["flux1"].flux
    T2 = sim_data["flux2"].flux
    a =int(len(T1)/2)
    T1 = T1[a] 
    T2 = T2[a]

    par = abs(T1-T2) # si es cero, salen de ambas guias la misma potencia
    return par


def create_3D_MMI_simulation(Len_MMI,MMI_width, wg_array_thickness, wg_array_width,wvg_length, gap, taper_length, freq0, fwidth, sin, sio2,freqs,len_corner,balance_weight,loss_weight):
    MMI_body = td.Structure(
    geometry = create_ridge2(MMI_width,0,-Len_MMI/2,Len_MMI/2,-wg_array_thickness/2,wg_array_thickness/2,0,len_corner),
    medium = sin,)

    Wg_in0 = td.Structure(
        geometry = create_ridge(wg_array_width,-gap/2,-(Len_MMI/2+taper_length+wvg_length),-(Len_MMI/2+taper_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_in0 = td.Structure(
        geometry = create_taper(MMI_width/3,-gap/2,-(Len_MMI/2+taper_length),-(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    Wg_in1 = td.Structure(
        geometry = create_ridge(wg_array_width,gap/2,-(Len_MMI/2+taper_length+wvg_length),-(Len_MMI/2+taper_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_in1 = td.Structure(
        geometry = create_taper(MMI_width/3,gap/2,-(Len_MMI/2+taper_length),-(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    Wg_out0 = td.Structure(
        geometry = create_ridge(wg_array_width,-gap/2,(Len_MMI/2+taper_length),(Len_MMI/2+taper_length+wvg_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_out0 = td.Structure(
        geometry = create_taper(MMI_width/3,-gap/2,(Len_MMI/2+taper_length),(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    Wg_out1 = td.Structure(
        geometry = create_ridge(wg_array_width,gap/2,(Len_MMI/2+taper_length),(Len_MMI/2+taper_length+wvg_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_out1 = td.Structure(
        geometry = create_taper(MMI_width/3,gap/2,(Len_MMI/2+taper_length),(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    ####

    #definimos los monitores y fuentes

    mode_spec = td.ModeSpec(
        num_modes=2,
        target_neff=3,
        track_freq="central",
        precision= "double",
        group_index_step=True
    )
    mode_source = td.ModeSource(
        center = (-gap/2,-(Len_MMI/2+taper_length+1.5),0),
        size = (3 * wg_array_width,0 , 5*wg_array_thickness),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction = "+",
        mode_spec = mode_spec,
        mode_index=0,
        num_freqs=5,
    )
    field_monitor1 = td.FieldMonitor(
        center = (0,0,0), size = (td.inf,td.inf,0), freqs=[freq0], name = "field1"
    )
    field_monitor11 = td.FieldMonitor(
        center = (0,0,wg_array_thickness/2), size = (td.inf,td.inf,0), freqs=[freq0], name = "field11"
    )
    field_monitor2 = td.FieldMonitor(
        center = (0,0,0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field2"
    )
    field_monitor3 = td.FieldMonitor(
        center = (0,Len_MMI/4,0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field3"
    )
    field_monitor4 = td.FieldMonitor(
        center = (0,-Len_MMI/4,0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field4"
    )

    field_monitor5 = td.FieldMonitor(
        center = (0,(Len_MMI/2+taper_length+1.5),0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field5"
    )

    flux_monitor0 = td.FluxMonitor(
        center = (-gap/2,-(Len_MMI/2+taper_length+1.4),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux0",
    )


    flux_monitor1 = td.FluxMonitor(
        center = (-gap/2,(Len_MMI/2+taper_length+1.5),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux1",
    )

    mode_monitor1 = td.ModeMonitor(
        center = (-gap/2, (Len_MMI/2+taper_length+3/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode1",

    )

    flux_monitor2 = td.FluxMonitor(
        center = (gap/2,(Len_MMI/2+taper_length+1.5),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux2",
    )

    mode_monitor2 = td.ModeMonitor( 
        center = (gap/2, (Len_MMI/2+taper_length+3/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode2",
    )
    flux_monitor00 = td.FluxMonitor(
        center = (-gap/2,-(Len_MMI/2),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux00",
    )

    mode_monitor00 = td.ModeMonitor(
        center = (-gap/2, -(Len_MMI/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode00",

    )

    flux_monitor11 = td.FluxMonitor(
        center = (-gap/2,(Len_MMI/2),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux11",
    )

    mode_monitor11 = td.ModeMonitor(
        center = (-gap/2, (Len_MMI/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode11",

    )

    flux_monitor22 = td.FluxMonitor(
        center = (gap/2,(Len_MMI/2),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux22",
    )

    mode_monitor22 = td.ModeMonitor( 
        center = (gap/2, (Len_MMI/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode22",
    )

    Lx = 1.5*MMI_width 
    Ly= 1.5*(Len_MMI+taper_length) 
    Lz = 2*wg_array_thickness
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=15,wavelength=1.55)
    sim = td.Simulation(
        size = (Lx,Ly,Lz),
        grid_spec = grid_spec,
        run_time = 3e-12,
        boundary_spec= td.BoundarySpec.all_sides(boundary = td.PML()),
        medium=sio2,
        structures=(MMI_body,Wg_in0,Taper_in0,Wg_in1,Taper_in1,Wg_out0,Taper_out0,Wg_out1,Taper_out1),
        sources=[mode_source],
        monitors = [field_monitor1,field_monitor2,field_monitor3,field_monitor4,field_monitor5,flux_monitor0,flux_monitor1,flux_monitor2]#,mode_monitor1,mode_monitor2,flux_monitor00,flux_monitor11,flux_monitor22,mode_monitor00,mode_monitor11,mode_monitor22],
    )
    job = web.Job(simulation=sim, task_name="MMI_2x2", verbose=True)
    sim_data = job.run(path="data/MMI_2x2.hdf5")

    T1 = sim_data["flux1"].flux
    T2 = sim_data["flux2"].flux

    a =int(len(T1)/2)
    T1 = T1[a] 
    T2 = T2[a]
    
    par = balance_weight*(abs(T1-T2)) + loss_weight*(1 -T1 -T2)
    return par

def create_2D_MMI_simulation_only(Len_MMI_array):
    input_positions = [-1/6,1/6] #posiciones de las waveguides de entrada
    wg_array_width=1.00 #anchura de la waveguide de entrada/salida
    wg_array_thickness=0.8 #altura de la waveguide de entrada/salida, es igual que la del MMI
    MMI_width=6.0 #anchura del MMI
    gap = (input_positions[1]-input_positions[0]) * MMI_width 
    wvlenth = np.linspace(1.5,1.6,101)
    freqs = td.C_0 / wvlenth
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))
    freq0 = td.C_0 / 1.55
    wvg_length = 1000
    taper_length = 5.0
    len_corner = taper_length
    #en primer lugar, definimos los materiales
    Len_MMI = Len_MMI_array[0] #longitud del MMI
    #sin = td.material_library['SiN']['Horiba'] #cristaline silicon
    #sio2 = td.material_library['SiO2']['Horiba']

    n_sin = 1.99
    n_sio2 = 1.44

    sin = td.Medium(permittivity=n_sin**2)
    sio2 = td.Medium(permittivity=n_sio2**2)
        
        
    
    MMI_body = td.Structure(
    geometry = create_ridge2(MMI_width,0,-Len_MMI/2,Len_MMI/2,-wg_array_thickness/2,wg_array_thickness/2,0,len_corner),
    medium = sin,)

    Wg_in0 = td.Structure(
        geometry = create_ridge(wg_array_width,-gap/2,-(Len_MMI/2+taper_length+wvg_length),-(Len_MMI/2+taper_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_in0 = td.Structure(
        geometry = create_taper(MMI_width/3,-gap/2,-(Len_MMI/2+taper_length),-(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    Wg_in1 = td.Structure(
        geometry = create_ridge(wg_array_width,gap/2,-(Len_MMI/2+taper_length+wvg_length),-(Len_MMI/2+taper_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_in1 = td.Structure(
        geometry = create_taper(MMI_width/3,gap/2,-(Len_MMI/2+taper_length),-(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    Wg_out0 = td.Structure(
        geometry = create_ridge(wg_array_width,-gap/2,(Len_MMI/2+taper_length),(Len_MMI/2+taper_length+wvg_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_out0 = td.Structure(
        geometry = create_taper(MMI_width/3,-gap/2,(Len_MMI/2+taper_length),(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    Wg_out1 = td.Structure(
        geometry = create_ridge(wg_array_width,gap/2,(Len_MMI/2+taper_length),(Len_MMI/2+taper_length+wvg_length),-wg_array_thickness/2,wg_array_thickness/2,0),
        medium = sin,
    )
    Taper_out1 = td.Structure(
        geometry = create_taper(MMI_width/3,gap/2,(Len_MMI/2+taper_length),(Len_MMI/2),-wg_array_thickness/2,wg_array_thickness/2,0,wg_array_width),
        medium = sin,
    )

    ####

    #definimos los monitores y fuentes

    mode_spec = td.ModeSpec(
        num_modes=2,
        target_neff=3,
        track_freq="central",
        precision= "double",
        group_index_step=True
    )
    mode_source = td.ModeSource(
        center = (-gap/2,-(Len_MMI/2+taper_length+1.5),0),
        size = (3 * wg_array_width,0 , 5*wg_array_thickness),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction = "+",
        mode_spec = mode_spec,
        mode_index=0,
        num_freqs=5,
    )
    field_monitor1 = td.FieldMonitor(
        center = (0,0,0), size = (td.inf,td.inf,0), freqs=[freq0], name = "field1"
    )
    field_monitor11 = td.FieldMonitor(
        center = (0,0,wg_array_thickness/2), size = (td.inf,td.inf,0), freqs=[freq0], name = "field11"
    )
    field_monitor2 = td.FieldMonitor(
        center = (0,0,0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field2"
    )
    field_monitor3 = td.FieldMonitor(
        center = (0,Len_MMI/4,0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field3"
    )
    field_monitor4 = td.FieldMonitor(
        center = (0,-Len_MMI/4,0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field4"
    )

    field_monitor5 = td.FieldMonitor(
        center = (0,(Len_MMI/2+taper_length+1.5),0), size = (td.inf,0,td.inf), freqs=[freq0], name = "field5"
    )

    flux_monitor0 = td.FluxMonitor(
        center = (-gap/2,-(Len_MMI/2+taper_length+1.4),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux0",
    )


    flux_monitor1 = td.FluxMonitor(
        center = (-gap/2,(Len_MMI/2+taper_length+1.5),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux1",
    )

    mode_monitor1 = td.ModeMonitor(
        center = (-gap/2, (Len_MMI/2+taper_length+3/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode1",

    )

    flux_monitor2 = td.FluxMonitor(
        center = (gap/2,(Len_MMI/2+taper_length+1.5),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux2",
    )

    mode_monitor2 = td.ModeMonitor( 
        center = (gap/2, (Len_MMI/2+taper_length+3/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode2",
    )
    flux_monitor00 = td.FluxMonitor(
        center = (-gap/2,-(Len_MMI/2),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux00",
    )

    mode_monitor00 = td.ModeMonitor(
        center = (-gap/2, -(Len_MMI/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode00",

    )

    flux_monitor11 = td.FluxMonitor(
        center = (-gap/2,(Len_MMI/2),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux11",
    )

    mode_monitor11 = td.ModeMonitor(
        center = (-gap/2, (Len_MMI/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode11",

    )

    flux_monitor22 = td.FluxMonitor(
        center = (gap/2,(Len_MMI/2),0),
        size = (2 * wg_array_width,0,5*wg_array_thickness),
        freqs = freqs,
        name = "flux22",
    )

    mode_monitor22 = td.ModeMonitor( 
        center = (gap/2, (Len_MMI/2),0),
        size = (2 * wg_array_width,0 , 5*wg_array_thickness),
        freqs= freqs,
        mode_spec = td.ModeSpec(num_modes=1,target_neff=3),
        name = "mode22",
    )

    Lx = 1.5*MMI_width 
    Ly= 1.5*(Len_MMI+taper_length) 
    Lz = 2*wg_array_thickness
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=15,wavelength=1.55)
    sim = td.Simulation(
        size = (Lx,Ly,Lz),
        grid_spec = grid_spec,
        run_time = 3e-12,
        boundary_spec= td.BoundarySpec.all_sides(boundary = td.PML()),
        medium=sio2,
        structures=(MMI_body,Wg_in0,Taper_in0,Wg_in1,Taper_in1,Wg_out0,Taper_out0,Wg_out1,Taper_out1),
        sources=[mode_source],
        monitors = [field_monitor1,field_monitor2,field_monitor3,field_monitor4,field_monitor5,flux_monitor0,flux_monitor1,flux_monitor2]#,mode_monitor1,mode_monitor2,flux_monitor00,flux_monitor11,flux_monitor22,mode_monitor00,mode_monitor11,mode_monitor22],
    )

    reference_point = (0,0)
    waveguide_point = (0,0)
    other_point = (0,(Len_MMI/2+taper_length+1.5))

    spectrum = wvlenth[::10]

    waveguide_medium = approximate_material(sim, waveguide_point, reference_point, spectrum)
    background_medium = approximate_material(sim, other_point, reference_point, spectrum)

    sim_2D = create_2D_sim(sim, [waveguide_medium, background_medium])
 # si es cero, salen de ambas guias la misma potencia
    return sim_2D