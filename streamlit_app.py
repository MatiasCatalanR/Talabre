import streamlit as st
import pandas as pd
import openpyxl
import altair as alt
import numpy as np
import locale
import math
import calendar 
import matplotlib.pyplot as plt
import requests
import io
import base64
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import datetime
from datetime import time
from datetime import datetime, timedelta
from streamlit_echarts import st_pyecharts
from pyecharts.charts import Gauge
from pyecharts import options as opts
from pyecharts.charts import Pie
from streamlit_extras.metric_cards import style_metric_cards 
from streamlit_echarts import st_echarts
from PIL import Image
from pandas.tseries.offsets import DateOffset
from openpyxl import load_workbook
#from st_circular_progress import CircularProgress
import requests
from requests.auth import HTTPBasicAuth





# Configurar la página
st.set_page_config(
    page_title="Reportabilidad UCO",
    page_icon="https://cdn-icons-png.flaticon.com/512/1170/1170667.png",
    layout="wide"
)



# Obtener la imagen de la URL
#image_url = "https://www.isdevelp.cl/wp-content/uploads/2022/05/LOGO-DEVELP-white-completo.png"
#response = requests.get(image_url)
#image = Image.open(io.BytesIO(response.content))

# Convertir la imagen a bytes
buf = io.BytesIO()
#image.save(buf, format='PNG')
image_bytes = buf.getvalue()

# Crear un bloque de HTML con la imagen centrada
#html = f'<img src="data:image/png;base64,{base64.b64encode(image_bytes).decode()}" style="display: block; margin: auto; width: 105%;">'

# Mostrar el HTML en la barra lateral
#st.sidebar.markdown(html, unsafe_allow_html=True)

# Agregar un separador después de la imagen

st.sidebar.markdown("<h1 style='text-align: center; color: white;'>Unidad de Control Operativo</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border:2.5px solid white'> </hr>", unsafe_allow_html=True)

funcion=st.sidebar.selectbox("Seleccione una Función",["Análisis Excel Avance IX Etapa","Reporte Inicio-Término Turno","En Desarrollo 1","En Desarrollo 2","Transgresiones de Velocidad","Equipos Obras Anexas","Pórtico","Tiempo en Geocercas"])

url_despacho='https://icons8.com/icon/21183/in-transit'





# Ahora 'data' contiene la respuesta de la API en formato JSON

if funcion=="Reporte Inicio-Término Turno":
    import datetime

    st.title('📊 Análisis de Relleno al Inicio y Término del Turno')
    # Obtener la fecha seleccionada por el usuario
    #selected_date = st.sidebar.date_input("Seleccione una fecha")

    dias_a_restar = 6



    # Obtén la fecha de hoy y la fecha de hace 7 días
    hoy = datetime.date.today()
    hace_siete_dias = hoy - datetime.timedelta(days=dias_a_restar)

    # Crea el selector de fechas con el rango predeterminado
    d = st.sidebar.date_input(
        "Seleccione una Fecha",
        (hace_siete_dias, hoy),
        format="MM.DD.YYYY",
    )



    # Restar días
    #nueva_fecha = selected_date - timedelta(days=dias_a_restar)
    #st.write(nueva_fecha)

    if len(d)==2:
        st.subheader("Este análisis contempla el estudio desde el "+str(d[0])+" hasta el "+str(d[1])) 
        mañana = d[1] + datetime.timedelta(days=1)

        # URL de la API
        url = "https://api.terrestra.tech/cycles?start_date="+str(d[0]) +" 08:00:00&end_date="+str(mañana)+" 08:00:00"
        #ajustar horometros en algun momento 
        url2 ="https://api.terrestra.tech/horometers?start_date="+str(d[0])+" 08:00:00&end_date="+str(d[0])+" 20:00:00"
    else:
        mañana = d[0] + datetime.timedelta(days=1)
        url = "https://api.terrestra.tech/cycles?start_date="+str(d[0]) +" 08:00:00&end_date="+str(mañana)+" 08:00:00"
        url2 ="https://api.terrestra.tech/horometers?start_date="+str(d[0])+" 08:00:00&end_date="+str(mañana)+" 08:00:00"
    # Credenciales 
    username = "talabre"
    password = "cosmos_talabre_2024"

    # Realizar la petición GET a la API
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    # Verificar si la petición fue exitosa 

    response2 = requests.get(url2, auth=HTTPBasicAuth(username, password))
    boton='off'
    if response.status_code == 200 and response2.status_code == 200:
        # Convertir la respuesta a JSON y df
        data = response.json()
        df_ciclo=pd.DataFrame(data)
        data2 = response2.json()
        df_horometro=pd.DataFrame(data2)
        #st.write(df_horometro)
        
        if st.sidebar.toggle('Mostrar base de datos'):
            boton='on'
            st.write("Reporte Ciclos")
            st.write(df_ciclo)
            st.write("Reporte Horometro")
            st.write(df_horometro)
        


    

        st.header("Resumen de KPIs")

        df=df_ciclo
        df['carga_teorica__m3_'] = pd.to_numeric(df['carga_teorica__m3_'], errors='coerce')
        df['distancia_recorrida__km_']=df['distancia_recorrida__km_'].str.replace(',', '.').astype(float)
        df['distancia_recorrida__km_'] = pd.to_numeric(df['distancia_recorrida__km_'])
        m3_transportados = df['carga_teorica__m3_'].sum()
        total_ciclos=len(df)
        km_recorridos = df['distancia_recorrida__km_'].sum()
        camiones_totales=df['patente'].nunique()
        col5,col6,col7=st.columns(3)
        col5.metric(label="m³ Transportados", value=format(total_ciclos*20, ',').replace(',', '.'))
        col6.metric(label="Total Ciclos", value=format(total_ciclos, ',').replace(',', '.'))
        col7.metric(label="Camiones Operativos", value=format(camiones_totales, ',').replace(',', '.'))
        st.metric(label="m³ Geométricos", value=format(int((total_ciclos*14)), ',').replace(',', '.'))

        #st.metric(label="Km Totales",value=int(km_recorridos))


        # Llamar a la función style_metric_cards() antes de crear la tarjeta métrica
        style_metric_cards()
        col8,col9=st.columns(2)
        # Agrupamos por 'lugar_descarga' y sumamos 'carga_teorica__m3_'
        data_carga = df.groupby('lugar_carga')['carga_teorica__m3_'].sum()

        # Creamos una lista de diccionarios para la opción de datos en la serie
        data_list_c = [{"value": v, "name": n} for n, v in data_carga.items()]
        data_list_c = sorted(data_list_c, key=lambda x: x["value"],reverse=True)


        options = {
            "tooltip": {"trigger": "item"},
            "legend": {"show": False},
            "series": [
                {
                    "name": "Origen Carga",
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": False,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    },
                    "label": {
                        "show": True,
                        "position": "outside",
                        "formatter": "{b}: {c} ({d}%)"
                    },
                    "emphasis": {
                        "label": {"show": True, "fontSize": "20", "fontWeight": "bold"}
                    },
                    "labelLine": {"show": True},
                    "data": data_list_c,
                    "color": ["#bb5726","#76151f","#f4a700","#e96c28","#209eb0","#374752","#c8b499","#004c4e","#6e4d48"]
                }
            ],
        }


        with col8:
            st.markdown("**Metros Cúbicos por Origen**")
            st_echarts(options=options, height="300px")


        data_descarga = df.groupby('lugar_descarga')['carga_teorica__m3_'].sum()

        # Creamos una lista de diccionarios para la opción de datos en la serie
        data_list_d = [{"value": v, "name": n} for n, v in data_descarga.items()]
        data_list_d = sorted(data_list_d, key=lambda x: x["value"],reverse=True)

        optionsd = {
            "tooltip": {"trigger": "item"},
            "legend": {"show": False},
            "series": [
                {
                    "name": "Destino Descarga",
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": False,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    },
                    "label": {
                        "show": True,
                        "position": "outside",
                        "formatter": "{b}: {c} ({d}%)"
                    },
                    "emphasis": {
                        "label": {"show": True, "fontSize": "20", "fontWeight": "bold"}
                    },
                    "labelLine": {"show": True},
                    "data": data_list_d,
                    "color": ["#374752","#c8b499","#004c4e","#6e4d48","#bb5726","#76151f","#f4a700","#e96c28","#209eb0"]
                }
            ],
        }

        with col9:
            st.markdown("**Metros Cúbicos por Destino**")
            st_echarts(options=optionsd, height="300px")

        def corregir_tiempo(x):
            if x.endswith('-04:00'):
                return pd.to_datetime(x).tz_localize(None)
            elif x.endswith('-03:00'):
                return pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
            else:
                return pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')

        df['fin_descarga'] = df['fin_descarga'].apply(corregir_tiempo)
        df['inicio_ciclo'] = df['inicio_ciclo'].apply(corregir_tiempo)
        df['fin_carga'] = df['fin_carga'].apply(corregir_tiempo)
        df['entrada_carguio'] = df['entrada_carguio'].apply(corregir_tiempo)
        df['inicio_carga'] = df['inicio_carga'].apply(corregir_tiempo)

        # Convertir 'fin_descarga' a datetime y redondear a la hora más cercana
        df['fin_descarga'] = pd.to_datetime(df['fin_descarga'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['fin_descarga'] = pd.to_datetime(df['fin_descarga']).dt.round('H')

        # Agrupar por 'lugar_descarga' y 'fin_descarga', y contar el número de entradas
        df_grouped = df.groupby(['lugar_descarga', 'fin_descarga']).size().reset_index(name='count')

        # Calcular los acumulados para cada 'lugar_descarga'
        df_grouped['count_acumulado'] = df_grouped.groupby('lugar_descarga')['count'].cumsum()

        # Crear un nuevo DataFrame que tenga un registro para cada hora del día para cada 'lugar_descarga'
        all_hours = pd.date_range(start=df['fin_descarga'].min(), end=df['fin_descarga'].max(), freq='H')
        all_lugares = df['lugar_descarga'].unique()
        df_all = pd.DataFrame([(lugar, hour) for lugar in all_lugares for hour in all_hours], columns=['lugar_descarga', 'fin_descarga'])

        # Llenar el nuevo DataFrame con los datos acumulados del DataFrame original
        df_all = df_all.merge(df_grouped, on=['lugar_descarga', 'fin_descarga'], how='left')

        # Rellenar los valores NaN con el valor anterior en la columna 'count_acumulado' dentro de cada grupo de 'lugar_descarga'
        df_all['count_acumulado'] = df_all.groupby('lugar_descarga')['count_acumulado'].fillna(method='ffill')

        # Rellenar los valores NaN restantes (es decir, los que están al principio de cada grupo de 'lugar_descarga') con 0
        df_all['count_acumulado'].fillna(0, inplace=True)
        df_all.sort_values('fin_descarga')
        #st.write(df_all)

        # Crear el gráfico de área apilada
        options = {
            "title": {"text": "Producción Acumulada"},
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "cross", "label": {"backgroundColor": "#6a7985"}},
            },
            "legend": {"data": [lugar for lugar in all_lugares]},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": [
                {
                    "type": "category",
                    "boundaryGap": False,
                    "data": df_all[df_all['count_acumulado'].notna()]['fin_descarga'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                }
            ],
            "yAxis": [{"type": "value"}],
            "series": [
                {
                    "name": lugar,
                    "type": "line",
                    "stack": "total",
                    "areaStyle": {},
                    "emphasis": {"focus": "series"},
                    "data": df_all[df_all['lugar_descarga'] == lugar]['count_acumulado'].tolist(),
                } for lugar in all_lugares
            ],
        }
        #st_echarts(options=options, height="400px")










        m3_km_transportado=int(m3_transportados)/int(km_recorridos)
        df['kph_mean_ruta']=df['kph_mean_ruta'].str.replace(',', '.').astype(float)
        df['kph_mean_ruta'] = pd.to_numeric(df['kph_mean_ruta'])
        df_velocidad = df[df['kph_mean_ruta'] > 15]

        #st.write(df_velocidad)
        velocidad_promedio=round(df_velocidad['kph_mean_ruta'].mean(),2)
        #st.write(velocidad_promedio)
        option_v = {
            "tooltip": {
                "formatter": '{a} <br/>{b} : {c}%'
            },
            "series": [{
                "name": velocidad_promedio,
                "type": 'gauge',
                "startAngle": 180,
                "endAngle": 0,
                "progress": {
                    "show": "true"
                },
                "radius":'100%', 
                "itemStyle": {
                    "color": '#76151f',
                    "shadowColor": 'rgba(128,128,128,0.45)',

                    "shadowBlur": 10,
                    "shadowOffsetX": 2,
                    "shadowOffsetY": 2,
                    "radius": '55%',
                },
                "progress": {
                    "show": "true",
                    "roundCap": "true",
                    "width": 15
                },
                "pointer": {
                    "length": '60%',
                    "width": 8,
                    "offsetCenter": [0, '5%']
                },
                "detail": {
                    "valueAnimation": "true",
                    "formatter": '{value}',
                    "backgroundColor": '#c8b499',
                    "borderColor": '#999',
                    "borderWidth": 4,
                    "width": '60%',
                    "lineHeight": 20,
                    "height": 20,
                    "borderRadius": 188,
                    "offsetCenter": [0, '40%'],
                    "valueAnimation": "true",
                },
                "data": [{
                    "value": velocidad_promedio,  # Calculamos el porcentaje
                    "name": 'Velocidad Promedio Flota [Km/Hr]'
                }]
            }]
        }

        st_echarts(options=option_v)


        # Suponiendo que 'df' es tu DataFrame y que las columnas relevantes son 'fecha', 'patente', 'inicio_ciclo' y 'fin_carga'

        # Convertir las columnas de tiempo a datetime
        df['inicio_ciclo'] = pd.to_datetime(df['inicio_ciclo'])
        df['fecha']=df['inicio_ciclo'].dt.date
        df['inicio_ciclo_hora']=df['inicio_ciclo'].dt.time
        df['fin_carga'] = pd.to_datetime(df['fin_carga'])
        df['fin_carga_hora']=df['fin_carga'].dt.time
        


        # Definir las horas de inicio y fin de los turnos diurno y nocturno
        inicio_diurno = pd.Timestamp('08:00:00').time()
        fin_diurno = pd.Timestamp('20:00:00').time()
        inicio_nocturno = pd.Timestamp('20:00:00').time()
        fin_nocturno = pd.Timestamp('08:00:00').time()

        # Crear DataFrames vacíos para los turnos diurno y nocturno
        turno_diurno = []
        turno_nocturno = []

        # Procesar los datos para cada día y patente
        for (fecha, patente), group in df.groupby(['fecha', 'patente']):
            # Turno diurno
            inicio_turno_diurno = group.loc[(group['inicio_ciclo_hora'] >= inicio_diurno) & (group['inicio_ciclo_hora'] < fin_diurno), 'inicio_ciclo_hora'].min()
            fin_turno_diurno = group.loc[(group['fin_carga_hora'] > inicio_diurno) & (group['fin_carga_hora'] <= fin_diurno), 'fin_carga_hora'].max()
            #fin_turno_diurno = group.loc[(group['fin_carga_hora'] >= inicio_diurno) & (group['fin_carga_hora'] < fin_diurno), 'fin_carga_hora'].min()
            # Turno nocturno
            inicio_turno_nocturno = group.loc[(group['inicio_ciclo_hora'] >= inicio_nocturno) | (group['inicio_ciclo_hora'] < fin_nocturno), 'inicio_ciclo_hora'].max()
            fin_turno_nocturno = group.loc[(group['fin_carga_hora'] > inicio_nocturno) | (group['fin_carga_hora'] <= fin_nocturno), 'fin_carga_hora'].min()
            
            # Si el turno nocturno termina al día siguiente
            #if fin_turno_nocturno < inicio_turno_nocturno:
                #st.write(fin_turno_nocturno)
                #fin_turno_nocturno += pd.Timedelta(days=1)
            
            # Añadir los resultados a las listas correspondientes
            turno_diurno.append({'fecha': fecha, 'patente': patente, 'inicio_turno': inicio_turno_diurno, 'fin_turno': fin_turno_diurno})
            turno_nocturno.append({'fecha': fecha, 'patente': patente, 'inicio_turno': inicio_turno_nocturno, 'fin_turno': fin_turno_nocturno})

        # Convertir las listas a DataFrames
        turno_diurno_df = pd.DataFrame(turno_diurno)
        turno_nocturno_df = pd.DataFrame(turno_nocturno)

        # Print the DataFrames



        # Print the DataFrames


        # Suponiendo que 'df' es tu DataFrame y que las columnas relevantes son 'fecha', 'patente', 'inicio_ciclo', 'entrada_carguio', 'fin_carga'

        # Convertir las columnas de tiempo a datetime
        df['inicio_ciclo'] = pd.to_datetime(df_ciclo['inicio_ciclo'])
        df['entrada_carguio'] = pd.to_datetime(df['entrada_carguio'])
        df['inicio_carga'] = pd.to_datetime(df['inicio_carga'])
        df['fin_carga'] = pd.to_datetime(df['fin_carga'])
        df['fecha']=df['inicio_ciclo'].dt.date

        # Definir las horas de inicio y fin del turno diurno
        inicio_diurno = pd.Timestamp('08:00:00').time()
        fin_diurno = pd.Timestamp('20:00:00').time()

        # Crear DataFrame vacío para el turno diurno
        turno_carguio = []
        fin_carga=[]
        inicio_carguio=[]
        inicio_diurno_time = pd.Timestamp('08:00:00').time()
        fin_diurno_time = pd.Timestamp('20:00:00').time()
        # Procesar los datos para cada día y patente

        df_carguio = df.dropna(subset=['entrada_carguio'])
        df_inicio_carga = df.dropna(subset=['inicio_carga'])
        #st.write(df_carguio)
        for (fecha, patente,tiempo_carga), group in df_carguio.groupby(['fecha', 'patente','tiempo_carga_min']):
            # Turno diurno
            entrada_carguio_diurno = group.loc[(group['entrada_carguio'].dt.time >= inicio_diurno_time) & (group['entrada_carguio'].dt.time < fin_diurno_time), 'entrada_carguio'].min()
            
            # Añadir los resultados a la lista correspondiente
            turno_carguio.append({'fecha': fecha, 'patente': patente, 'entrada_carguio': entrada_carguio_diurno, 'tiempo_carga':tiempo_carga})
        for (fecha, patente), group in df_inicio_carga.groupby(['fecha', 'patente']):
            # Turno diurno
            inicio_carga_diurno = group.loc[(group['inicio_carga'].dt.time >= inicio_diurno_time) & (group['inicio_carga'].dt.time < fin_diurno_time), 'inicio_carga'].min()
            
            # Añadir los resultados a la lista correspondiente
            inicio_carguio.append({'fecha': fecha, 'patente': patente, 'inicio_carga': inicio_carga_diurno})

        df_fin_carga=df.dropna(subset=['fin_carga'])
        for (fecha, patente), group in df_fin_carga.groupby(['fecha', 'patente']):
            # Turno diurno
            #fin_carga_diurno = group.loc[(group['fin_carga'].dt.time > inicio_diurno) & (group['fin_carga'].dt.time <= fin_diurno), 'fin_carga'].max()
            fin_carga_diurno = group.loc[(group['fin_carga'].dt.time >= inicio_diurno) & (group['fin_carga'].dt.time < fin_diurno), 'fin_carga'].min()
            
            # Añadir los resultados a la lista correspondiente
            fin_carga.append({'fecha': fecha, 'patente': patente,'fin_carga': fin_carga_diurno})
        
        
        

        # Convertir la lista a DataFrame
        entrada_carguio_df = pd.DataFrame(turno_carguio)
        inicio_carga_df = pd.DataFrame(inicio_carguio)

        fin_carga_df = pd.DataFrame(fin_carga)
        # Elimina las filas donde alguna columna es None
        turno_diurno_df = turno_diurno_df.dropna()
        turno_diurno_it=turno_diurno_df
        # Filtrar el DataFrame por el rango de horas
        turno_diurno_df = turno_diurno_df[(turno_diurno_df['inicio_turno'] >= time(8, 15, 0)) & (turno_diurno_df['inicio_turno'] <= time(11, 0, 0))]
        fin_turno_diurno_df = turno_diurno_it[(turno_diurno_it['fin_turno'] >= time(17, 0, 0)) & (turno_diurno_it['fin_turno'] <= time(20, 0, 0))]
        if fin_turno_diurno_df.empty:
            fin_turno_diurno_df = turno_diurno_it
            
        if turno_diurno_df.empty:
            turno_diurno_df = turno_diurno_it



        # Convertir 'inicio_turno' a segundos
        #st.write(turno_diurno_df)
        
        turno_diurno_df['inicio_turno_segundos'] = turno_diurno_df['inicio_turno'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        # Contabilizar los registros con el valor True en la columna "cargado_inicio_turno"
        inicio_cargado = (df['cargado_inicio_turno'] == 'True').sum()
        # Mostrar el valor como un KPI en Streamlit
        st.markdown(f'<h1 style="text-align: center;">{inicio_cargado}</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 20px;">Total Camiones Cargados al Inicio de Turno</p>', unsafe_allow_html=True)
        st.header("Turno Diurno:")
        # Calcular el promedio, mínimo y máximo
        promedio_segundos = turno_diurno_df['inicio_turno_segundos'].mean()
        minimo_segundos = turno_diurno_df['inicio_turno_segundos'].min()
        maximo_segundos = turno_diurno_df['inicio_turno_segundos'].max()
        dsv_segundos = turno_diurno_df['inicio_turno_segundos'].std()   

        # Convertir los segundos de vuelta a formato de hora
        promedio = str(np.timedelta64(int(promedio_segundos), 's'))
        minimo = str(np.timedelta64(int(minimo_segundos), 's'))
        maximo = str(np.timedelta64(int(maximo_segundos), 's'))
        dsv = str(np.timedelta64(int(dsv_segundos), 's'))

        # Convertir los segundos de vuelta a formato de hora
        promedio = str(timedelta(seconds=int(promedio_segundos)))
        minimo = str(timedelta(seconds=int(minimo_segundos)))
        maximo = str(timedelta(seconds=int(maximo_segundos)))
        dsv = str(timedelta(seconds=int(dsv_segundos))) 
        col1, col5,col2,col3,col4=st.columns(5)
        # Convertir 'inicio_diurno' a segundos
        inicio_diurno_segundos = inicio_diurno.hour * 3600 + inicio_diurno.minute * 60 + inicio_diurno.second

        # Calcular la diferencia entre el promedio y el inicio diurno
        diferencia_segundos = promedio_segundos - inicio_diurno_segundos

        # Convertir los segundos de vuelta a formato de hora
        diferencia = str(timedelta(seconds=int(abs(diferencia_segundos))))

            
        with col1:
            st.header("Inicio Turno")
            
            # Usar HTML y CSS para personalizar los colores
            st.markdown(f'<div style="color: blue; font-size: medium; padding: 10px; background-color: lightblue; border-radius: 10px;">Promedio: {promedio} </div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: green; font-size: medium; padding: 10px; background-color: lightgreen; border-radius: 10px;">Primer Camión: {minimo}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: red; font-size: medium; padding: 10px; background-color: lightcoral; border-radius: 10px;">Último Camión: {maximo}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: orange; font-size: medium; padding: 10px; background-color: lightyellow; border-radius: 10px;">Desviación Estándar: {dsv}</div>', unsafe_allow_html=True)
            inicio_turno_df = turno_diurno_df[['fecha', 'patente', 'inicio_turno']].copy()
            st.write(inicio_turno_df)
        # Agregar un gráfico para visualizar las tendencias
        #st.line_chart(turno_diurno_df['inicio_turno_segundos'])


        entrada_carguio_df['inicio_turno_segundos_carguio'] = entrada_carguio_df['entrada_carguio'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        #st.write(df_carguio)
        promedio_segundos =entrada_carguio_df['inicio_turno_segundos_carguio'].mean()
        minimo_segundos = entrada_carguio_df['inicio_turno_segundos_carguio'].min()
        maximo_segundos = entrada_carguio_df['inicio_turno_segundos_carguio'].max()
        dsv_segundos = entrada_carguio_df['inicio_turno_segundos_carguio'].std()   
        # Convertir los segundos de vuelta a formato de hora
        promedio = str(np.timedelta64(int(promedio_segundos), 's'))
        minimo = str(np.timedelta64(int(minimo_segundos), 's'))
        maximo = str(np.timedelta64(int(maximo_segundos), 's'))
        dsv = str(np.timedelta64(int(dsv_segundos), 's'))
        # Convertir los segundos de vuelta a formato de hora
        promedio = str(timedelta(seconds=int(promedio_segundos)))
        minimo = str(timedelta(seconds=int(minimo_segundos)))
        maximo = str(timedelta(seconds=int(maximo_segundos))) 
        dsv = str(timedelta(seconds=int(dsv_segundos))) 
        diferencia_segundos = promedio_segundos - inicio_diurno_segundos
        diferencia = str(timedelta(seconds=int(abs(diferencia_segundos))))

        with col2:
            st.header("Entrada Carga")
            st.markdown(f'<div style="color: blue; font-size: medium; padding: 10px; background-color: lightblue; border-radius: 10px;">Promedio: {promedio} </div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: green; font-size: medium; padding: 10px; background-color: lightgreen; border-radius: 10px;">Primer Camión: {minimo}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: red; font-size: medium; padding: 10px; background-color: lightcoral; border-radius: 10px;">Último Camión: {maximo}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: orange; font-size: medium; padding: 10px; background-color: lightyellow; border-radius: 10px;">Desviación Estándar: {dsv}</div>', unsafe_allow_html=True)
            entrada_carga_df=entrada_carguio_df[['fecha','patente','entrada_carguio','tiempo_carga']].copy()
            st.write(entrada_carga_df)
            #iniciocarga
        inicio_carga_df['inicio_turno_segundos_carguio'] = inicio_carga_df['inicio_carga'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        #st.write(df_carguio)
        promedio_segundos =inicio_carga_df['inicio_turno_segundos_carguio'].mean()
        minimo_segundos = inicio_carga_df['inicio_turno_segundos_carguio'].min()
        maximo_segundos = inicio_carga_df['inicio_turno_segundos_carguio'].max()   
        dsv_segundos = inicio_carga_df['inicio_turno_segundos_carguio'].std()   

        # Convertir los segundos de vuelta a formato de hora
        promedio = str(np.timedelta64(int(promedio_segundos), 's'))
        minimo = str(np.timedelta64(int(minimo_segundos), 's'))
        maximo = str(np.timedelta64(int(maximo_segundos), 's'))
        dsv = str(np.timedelta64(int(dsv_segundos), 's'))

        # Convertir los segundos de vuelta a formato de hora
        promedio = str(timedelta(seconds=int(promedio_segundos)))
        minimo = str(timedelta(seconds=int(minimo_segundos)))
        maximo = str(timedelta(seconds=int(maximo_segundos))) 
        dsv = str(timedelta(seconds=int(dsv_segundos))) 

        diferencia_segundos = promedio_segundos - inicio_diurno_segundos
        diferencia = str(timedelta(seconds=int(abs(diferencia_segundos))))
        with col3:
            st.header("Inicio Carga")
            st.markdown(f'<div style="color: blue; font-size: medium; padding: 10px; background-color: lightblue; border-radius: 10px;">Promedio: {promedio} </div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: green; font-size: medium; padding: 10px; background-color: lightgreen; border-radius: 10px;">Primer Camión: {minimo}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: red; font-size: medium; padding: 10px; background-color: lightcoral; border-radius: 10px;">Último Camión: {maximo}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: orange; font-size: medium; padding: 10px; background-color: lightyellow; border-radius: 10px;">Desviación Estándar: {dsv}</div>', unsafe_allow_html=True)
            inicio_carga_df_df=inicio_carga_df[['fecha','patente','inicio_carga']].copy()
            
            st.write(inicio_carga_df)

        #inicio carga
        fin_carga_df['fin_carga_segundos'] = fin_carga_df['fin_carga'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        #st.write(df_carguio)
        promedio_segundos =fin_carga_df['fin_carga_segundos'].mean()
        minimo_segundos = fin_carga_df['fin_carga_segundos'].min()
        maximo_segundos = fin_carga_df['fin_carga_segundos'].max() 
        dsv_segundos = fin_carga_df['fin_carga_segundos'].std()    
        # Convertir los segundos de vuelta a formato de hora
        promedio = str(np.timedelta64(int(promedio_segundos), 's'))
        minimo = str(np.timedelta64(int(minimo_segundos), 's'))
        maximo = str(np.timedelta64(int(maximo_segundos), 's'))
        dsv = str(np.timedelta64(int(dsv_segundos), 's'))
        # Convertir los segundos de vuelta a formato de hora
        promedio = str(timedelta(seconds=int(promedio_segundos)))
        minimo = str(timedelta(seconds=int(minimo_segundos)))
        maximo = str(timedelta(seconds=int(maximo_segundos))) 
        dsv = str(timedelta(seconds=int(dsv_segundos)))
        diferencia_segundos = promedio_segundos - inicio_diurno_segundos
        diferencia = str(timedelta(seconds=int(abs(diferencia_segundos))))
        with col4:
            st.header("Fin Carguío")
            st.markdown(f'<div style="color: blue; font-size: medium; padding: 10px; background-color: lightblue; border-radius: 10px;">Promedio: {promedio} </div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: green; font-size: medium; padding: 10px; background-color: lightgreen; border-radius: 10px;">Primer Camión: {minimo}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: red; font-size: medium; padding: 10px; background-color: lightcoral; border-radius: 10px;">Último Camión: {maximo}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: orange; font-size: medium; padding: 10px; background-color: lightyellow; border-radius: 10px;">Desviación Estándar: {dsv}</div>', unsafe_allow_html=True)
            fin_cargas_df=fin_carga_df[['fecha','patente','fin_carga']]

            st.write(fin_cargas_df)
        #inicio carga
        fin_turno_diurno_df['fin_turno_segundos'] = fin_turno_diurno_df['fin_turno'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        #st.write(df_carguio)
        promedio_segundos =fin_turno_diurno_df['fin_turno_segundos'].mean()
        minimo_segundos = fin_turno_diurno_df['fin_turno_segundos'].min()
        maximo_segundos = fin_turno_diurno_df['fin_turno_segundos'].max() 
        dsv_segundos = fin_turno_diurno_df['fin_turno_segundos'].std()    
        # Convertir los segundos de vuelta a formato de hora
        promedio = str(np.timedelta64(int(promedio_segundos), 's'))
        minimo = str(np.timedelta64(int(minimo_segundos), 's'))
        maximo = str(np.timedelta64(int(maximo_segundos), 's'))
        dsv = str(np.timedelta64(int(dsv_segundos), 's'))
        # Convertir los segundos de vuelta a formato de hora
        promedio = str(timedelta(seconds=int(promedio_segundos)))
        minimo = str(timedelta(seconds=int(minimo_segundos)))
        maximo = str(timedelta(seconds=int(maximo_segundos))) 
        dsv = str(timedelta(seconds=int(dsv_segundos)))
        diferencia_segundos = promedio_segundos - inicio_diurno_segundos
        diferencia = str(timedelta(seconds=int(abs(diferencia_segundos))))
        with col5:
            st.header("Fin Turno")
            st.markdown(f'<div style="color: blue; font-size: medium; padding: 10px; background-color: lightblue; border-radius: 10px;">Promedio: {promedio} </div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: red; font-size: medium; padding: 10px; background-color: lightcoral; border-radius: 10px;">Primer Camión: {minimo}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: green; font-size: medium; padding: 10px; background-color: lightgreen; border-radius: 10px;">Último Camión: {maximo}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: orange; font-size: medium; padding: 10px; background-color: lightyellow; border-radius: 10px;">Desviación Estándar: {dsv}</div>', unsafe_allow_html=True)
            #fin_cargas_df=fin_turno_diurno_df[['fecha','patente','fin_turno']].copy()
            fin_turno_diurno_df=fin_turno_diurno_df[['fecha','patente','fin_turno']]
            st.write(fin_turno_diurno_df)
        entrada_carguio_df['entrada_carguio']=entrada_carguio_df['entrada_carguio'].dt.time
        inicio_carga_df['inicio_carga']=inicio_carga_df['inicio_carga'].dt.time
        
        fin_carga_df['fin_carga']=fin_carga_df['fin_carga'].dt.time
        


        # Supongamos que ya tienes los DataFrames turno_diurno_df, entrada_carguio_df y fin_carga_df cargados con los datos
        #st.write(turno_diurno_df)


        # Renombrar las columnas para que coincidan con las que necesitamos en el gráfico
        turno_diurno_df = turno_diurno_df.rename(columns={'inicio_turno': 'hora', 'patente': 'Patente'})
        entrada_carguio_df = entrada_carguio_df.rename(columns={'entrada_carguio': 'hora', 'patente': 'Patente'})
        inicio_carga_df = inicio_carga_df.rename(columns={'inicio_carga': 'hora', 'patente': 'Patente'})

        fin_carga_df = fin_carga_df.rename(columns={'fin_carga': 'hora', 'patente': 'Patente'})

        # Crear una nueva columna para identificar el origen de los datos en el DataFrame combinado
        turno_diurno_df['Origen'] = 'Inicio Turno'
        entrada_carguio_df['Origen'] = 'Entrada Carguio'
        inicio_carga_df['Origen'] = 'Inicio Carga'
        fin_carga_df['Origen'] = 'Fin Carga'

        # Concatenar los DataFrames
        combined_df = pd.concat([turno_diurno_df, entrada_carguio_df,inicio_carga_df, fin_carga_df], ignore_index=True)
        combined_df = combined_df.dropna(subset=['hora'])

        # Convertir la columna 'hora' a formato de cadena de texto si no está en ese formato
        combined_df['hora'] = combined_df['hora'].astype(str)

        # Crear una nueva columna 'hora_formateada' para las horas
        combined_df['hora_formateada'] = combined_df['hora'].str[-8:]
        #st.write(combined_df)
        # Graficar con Altair

###pruebaas inicio turno
        combined_df2 = pd.concat([turno_diurno_df,inicio_carga_df], ignore_index=True)
        combined_df2 = combined_df2.dropna(subset=['hora'])

        # Convertir la columna 'hora' a formato de cadena de texto si no está en ese formato
        combined_df2['hora'] = combined_df2['hora'].astype(str)

        # Crear una nueva columna 'hora_formateada' para las horas
        combined_df2['hora_formateada'] = combined_df2['hora'].str[-8:]
        #st.write(combined_df)
        # Convertir la columna 'hora_formateada' a formato datetime
        combined_df2['hora_formateada'] = pd.to_datetime(combined_df2['hora_formateada'], format='%H:%M:%S')
        # Crear una máscara para filtrar las horas menores a las 12:00:00

        mask = (combined_df2['hora_formateada'].dt.hour < 12)

        # Aplicar la máscara al DataFrame
        filtered_df = combined_df2[mask]

        if filtered_df.empty:
            filtered_df=combined_df2


        # Graficar con Altair




        # Asegúrate de que la columna 'hora' esté en formato de tiempo correcto
        filtered_df['hora'] = pd.to_datetime(filtered_df['hora'], format='%H:%M:%S')

        # Convierte las horas a números (total de segundos desde la medianoche)
        filtered_df['hora_numerica'] = filtered_df['hora'].dt.hour * 3600 + filtered_df['hora'].dt.minute * 60 + filtered_df['hora'].dt.second

        # Agrega una columna con la hora en formato de cadena
        filtered_df['Hora'] = filtered_df['hora'].dt.strftime('%H:%M:%S')

        # Ordena el DataFrame por la columna 'hora' para asegurar la secuencia correcta
        filtered_df = filtered_df.sort_values('hora')

        # Crea el gráfico de puntos con Plotly
        fig = px.scatter(filtered_df, x='Patente', y='hora_numerica', color='Origen',
                        labels={'hora_numerica': 'Hora del día'},
                        category_orders={"hora_numerica": sorted(filtered_df['hora_numerica'].unique())},
                        hover_data={'Hora': True, 'hora_numerica': False})

        # Actualiza el formato del eje y para mostrar solo horas enteras y lo invierte
        fig.update_yaxes(tickvals=list(range(0, 24*3600, 3600)), ticktext=[f'{h}:00:00' for h in range(24)])
        # Agrega un título al gráfico
        fig.update_layout(title='Primer Registro al Comienzo del Turno Diurno por Patente')
        fig.update_layout(width=1000, height=500)
        # Muestra el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Crear los dataframes basados en el valor de la columna "Origen"
        df_analisis_iniciot = filtered_df.loc[filtered_df['Origen'] == 'Inicio Turno', ['fecha', 'Patente', 'hora']]
        df_analisis_carga = filtered_df.loc[filtered_df['Origen'] == 'Inicio Carga', ['fecha', 'Patente', 'hora']]


        # Convertir la columna 'hora' a formato datetime
        df_analisis_iniciot['hora'] = pd.to_datetime(df_analisis_iniciot['hora'])
        df_analisis_carga['hora'] = pd.to_datetime(df_analisis_carga['hora'])

        df_analisis_iniciot['hora'] = df_analisis_iniciot['hora'].dt.strftime('%H:%M:%S')
        df_analisis_carga['hora'] = df_analisis_carga['hora'].dt.strftime('%H:%M:%S')


        # Suponiendo que 'df_analisis_iniciot' y 'df_analisis_carga' ya están definidos y contienen una columna 'hora'

        # Convertir 'hora' a datetime y luego redondear al intervalo más cercano de 10 minutos
        df_analisis_iniciot['hora'] = pd.to_datetime(df_analisis_iniciot['hora']).dt.round('10T').dt.strftime('%H:%M')
        df_analisis_carga['hora'] = pd.to_datetime(df_analisis_carga['hora']).dt.round('10T').dt.strftime('%H:%M')

        # Crear los histogramas con los datos agrupados
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        df_analisis_iniciot['hora'].value_counts().sort_index().plot(kind='bar', edgecolor='black', color='#f4a700')
        plt.title('Histograma Inicio Turno')
        plt.xlabel('Hora')
        plt.ylabel('Frecuencia')

        plt.subplot(1, 2, 2)
        df_analisis_carga['hora'].value_counts().sort_index().plot(kind='bar', edgecolor='black', color='#f4a700')
        plt.title('Histograma Inicio Carga')
        plt.xlabel('Hora')
        plt.ylabel('Frecuencia')
        #df_analisis_carga
        plt.tight_layout()
        st.pyplot(plt)
        #entrada_carguio_df

        #st.write(df.head())

        #st.write(df.info())


        #st.write(df.describe())



        histograma = df[['t_cola_carga', 'tiempo_carga_min', 'transito_cargado_min', 't_cola_descarga', 'tiempo_descarga_min', 'transito_descargado_min', 'tiempo_ciclo_min', 'kph_mean','kph_mean_ruta', 'kph_max', 'distancia_recorrida__km_', 'tiempo_demoras_min']]
        # Set Seaborn style 
        sns.set_style("darkgrid") 

        # Asumiendo que df es tu DataFrame original
        df_limpieza = df.dropna()

        # Aplicar la conversión a cadenas de texto a todos los elementos del dataframe
        histograma = histograma.apply(lambda x: x.astype(str))

        # Aplicar el reemplazo de comas por puntos a todos los elementos del dataframe
        histograma = histograma.applymap(lambda x: x.replace(',', '.'))

        # Reemplazar 'None' y valores nulos con '0'
        histograma = histograma.replace('None', '0').fillna('0')
        # Reemplazar los porcentajes y convertir a números flotantes
        histograma = histograma.apply(lambda x: x.str.replace('%','').astype(np.float64))

        if boton=="on":

            st.header("Análisis exploratorio de los datos")



            # Identify numerical columns 
            numerical_columns = histograma.select_dtypes(include=["int64", "float64"]).columns 
            # Plot distribution of each numerical feature 
            plt.figure(figsize=(14, len(numerical_columns) * 3)) 
            for idx, feature in enumerate(numerical_columns, 1): 
                plt.subplot(len(numerical_columns), 2, idx) 
                sns.histplot(histograma[feature], kde=True) 
                plt.title(f"{feature} | Skewness: {round(histograma[feature].skew(), 2)}") 
            
            # Adjust layout and show plots 
            plt.tight_layout() 
            #plt.show() 
            st.pyplot(plt)
            st.markdown("**Estadística Básica**")
            st.write(histograma.describe())
            st.markdown("**Matriz de correlación**")
            st.write(histograma.corr())


###pruebas inicio turno 

    else:
        st.error("Fecha sin Datos")

   
if funcion=="Equipos Obras Anexas":
    st.title("🏗️ Equipos trabajando en Obras Anexas")
    import matplotlib.pyplot as plt
    import matplotlib
    df=pd.read_csv("https://raw.githubusercontent.com/MatiasCatalanR/Talabre/main/2bce8243-fe86-4678-a840-55d5eeee3dfe.csv")
   
    if df is not None:
        df['tiempo_encendido_total'] = df['tiempo_encendido_total'].str.replace(',', '.').astype(float)
    # Asegurándonos de que 'fecha_inicio' es datetime
    df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])

    # Agrupando por fecha y patente, y sumando las horas encendidas
    df_grouped = df.groupby([df['fecha_inicio'].dt.date, 'truckpatent'])['tiempo_encendido_total'].sum().reset_index()
    # Obtén el mapa de colores 'Inferno_r'
    cmap = plt.get_cmap('rocket')

    # Selecciona 5 colores de manera uniforme a lo largo del mapa de colores
    colors = [cmap(i) for i in np.linspace(0, 1,6)]
    # Convierte los colores RGBA a formato hexadecimal
    hex_colors = [matplotlib.colors.rgb2hex(color) for color in colors]

    # Crea un diccionario que mapea las patentes a los colores
    color_dict = {'EXCA-184': hex_colors[0], 'EXCA-684': hex_colors[1], 'RETRO-365': hex_colors[2], 'RETRO-628': hex_colors[3], 'SWKS81': hex_colors[4]}
    df_grouped = df_grouped.sort_values('truckpatent', ascending=False)

    fig = px.bar(df_grouped, x='fecha_inicio', y='tiempo_encendido_total', color='truckpatent', title='Horas encendidas 27 de Junio al 16 de Julio',
                labels={'tiempo_encendido_total':'Horas Operativas', 'fecha_inicio':'Fecha', 'truckPatent':'Patente'},
                color_discrete_map=color_dict)   
    #st.plotly_chart(fig, use_container_width=True)
    
    import plotly.express as px
    import plotly.graph_objects as go

    # Ordenamos el DataFrame por fecha
    df_grouped = df_grouped.sort_values(by='fecha_inicio')

    # Calculamos el promedio de horas operativas por fecha
    promedio_horas_operativas = df_grouped.groupby("fecha_inicio")['tiempo_encendido_total'].mean().reset_index()

    fig = px.bar(df_grouped, x='fecha_inicio', y='tiempo_encendido_total', color='truckpatent', title='Horas encendidas 27 de Junio al 16 de Julio',
                        labels={'tiempo_encendido_total':'Horas Operativas', 'fecha_inicio':'Fecha', 'truckpatent':'Patente'},
                        color_discrete_map=color_dict)

    fig.update_layout(barmode='group')

    # Agregamos una recta con el promedio de horas operativas por fecha
    fig.add_trace(go.Scatter(x=promedio_horas_operativas['fecha_inicio'], y=promedio_horas_operativas['tiempo_encendido_total'], mode='lines',line={'dash': 'dash','color':'black'}, name='Promedio Horas Operativas'))

    st.plotly_chart(fig, use_container_width=True)

    import datetime

    dias_a_restar = 6

    # Obtén la fecha de hoy y la fecha de hace 7 días
    hoy = datetime.date.today()
    hace_siete_dias = hoy - datetime.timedelta(days=dias_a_restar)

    # Crea el selector de fechas con el rango predeterminado
    d = st.sidebar.date_input(
        "Seleccione una Fecha",
        (hace_siete_dias, hoy),
        format="MM.DD.YYYY",
    )


    # Restar días
    #nueva_fecha = selected_date - timedelta(days=dias_a_restar)
    #st.write(nueva_fecha)

    if len(d)==2:
        mañana = d[1] + datetime.timedelta(days=1)

        # URL de la API
        #ajustar horometros en algun momento 
        url2 ="https://api.terrestra.tech/horometers?start_date="+str(d[0])+" 08:00:00&end_date="+str(mañana)+" 08:00:00"
    else:
        mañana = d[0] + datetime.timedelta(days=1)
        url2 ="https://api.terrestra.tech/horometers?start_date="+str(d[0])+" 08:00:00&end_date="+str(mañana)+" 08:00:00"
    # Credenciales 
    username = "talabre"
    password = "cosmos_talabre_2024"


    # Verificar si la petición fue exitosa 

    response2 = requests.get(url2, auth=HTTPBasicAuth(username, password))
    boton='off'
    if response2.status_code == 200:
        # Convertir la respuesta a JSON y df
        data2 = response2.json()
        df_horometro=pd.DataFrame(data2)
        #st.write(df_horometro)
        
        df=df_horometro
        geocercas = df['truckPatent'].unique()
        geocercas = np.insert(geocercas, 0, "Todas")
        patentes_anexas = [ 'EXCA-684', 'RETRO-365', 'RETRO-628', 'SWKS81']
        patentes_anexas = [patente for patente in patentes_anexas if patente in geocercas]
        selected_geocerca = st.sidebar.multiselect("Selecciona las Patentes", geocercas, default=patentes_anexas)
        if "Todas" in selected_geocerca:
            filtered_df = df  # No se aplica ningún filtro
        else:
            filtered_df = df[df['truckPatent'].isin(selected_geocerca)]
        df=filtered_df





        df["fecha_inicio"]=pd.to_datetime(df["fecha_inicio"])
        if df is not None:
            df['tiempo_encendido_total'] = df['tiempo_encendido_total'].str.replace(',', '.').astype(float)
        # Asegurándonos de que 'fecha_inicio' es datetime
        df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])

        # Agrupando por fecha y patente, y sumando las horas encendidas
        df_grouped = df.groupby([df['fecha_inicio'].dt.date, 'truckPatent'])['tiempo_encendido_total'].sum().reset_index()
        df_grouped = df_grouped.sort_values('truckPatent', ascending=False)

        fig = px.bar(df_grouped, x='fecha_inicio', y='tiempo_encendido_total', color='truckPatent', title='Horas encendidas Periodo-Patentes Seleccionadas',
                    labels={'tiempo_encendido_total':'Horas Operativas', 'fecha_inicio':'Fecha', 'truckPatent':'Patente'},
                    color_discrete_map=color_dict)

        fig.update_xaxes(type='category')  # Establecer el tipo de eje x como categoría para mostrar solo fechas
        fig.update_layout(xaxis={'type': 'category', 'categoryorder': 'category ascending'})  # Ordenar las fechas de forma ascendente
        promedio_horas_operativas = df_grouped.groupby("fecha_inicio")['tiempo_encendido_total'].mean().reset_index()

        #st.plotly_chart(fig, use_container_width=True)
        fig = px.bar(df_grouped, x='fecha_inicio', y='tiempo_encendido_total', color='truckPatent', title='Horas encendidas equipos y periodo seleccionado',
                        labels={'tiempo_encendido_total':'Horas Operativas', 'fecha_inicio':'Fecha', 'truckpatent':'Patente'},
                        color_discrete_map=color_dict)   

        fig.update_layout(barmode='group')

        # Agregamos una recta con el promedio de horas operativas por fecha
        fig.add_trace(go.Scatter(x=promedio_horas_operativas['fecha_inicio'], y=promedio_horas_operativas['tiempo_encendido_total'], mode='lines',line={'dash': 'dash','color':'black'}, name='Promedio Horas Operativas'))

        st.plotly_chart(fig, use_container_width=True)
        st.header("HH Periodo Seleccionado:")

        col2,col1=st.columns(2)
        style_metric_cards()
        with col1:
            st.metric(label="Promedio de horas Operativas", value=round(df_grouped['tiempo_encendido_total'].mean(),1))
        with col2:
            st.metric(label="Total de Horas Operativas", value=round(df_grouped['tiempo_encendido_total'].sum(),1))
    else:
        st.error("Seleccione un Periodo mas Acotado/con Datos en obras Anexas")
if funcion=="Transgresiones de Velocidad":
    st.title("🛑 Transgresiones de Velocidad")

    import pandas as pd
    from pyecharts import options as opts
    from pyecharts.charts import Pie
    from streamlit_echarts import st_pyecharts

    df=pd.read_csv("https://raw.githubusercontent.com/MatiasCatalanR/Talabre/main/Transgresiones_historicas%20(7).csv")
    
    if df is not None:
        df['exceso_velocidad_kmh'] = df['exceso_velocidad_kmh'].str.replace(',', '.').astype(float)
        df['velocidad_kmh'] = df['velocidad_kmh'].str.replace(',', '.').astype(float)

        geocercas = df['geocerca'].unique()
        geocercas = np.insert(geocercas, 0, "Todas")
        selected_geocerca = st.sidebar.multiselect("Selecciona las Geocercas", geocercas, default=["Todas"])
        # Agregar checkboxes y slider
        if "Todas" in selected_geocerca:
            filtered_df = df  # No se aplica ningún filtro
        else:
            filtered_df = df[df['geocerca'].isin(selected_geocerca)]
        
        st.sidebar.markdown("**Dirección:**")
        col1, col2 = st.sidebar.columns(2)
        direccion_subida = col1.checkbox("Subida", value=True)
        direccion_bajada = col2.checkbox("Bajada", value=True)
        if direccion_subida and direccion_bajada:
            nada=0
        else:
            if direccion_subida:
                filtered_df=filtered_df[filtered_df['velocidad_kmh']-filtered_df['exceso_velocidad_kmh']-filtered_df['limite_subida']==0]
            if direccion_bajada:
                filtered_df=filtered_df[filtered_df['velocidad_kmh']-filtered_df['exceso_velocidad_kmh']-filtered_df['limite_bajada']==0]
        exceso_velocidad = st.sidebar.slider("Excesos de Velocidad (km/hr)", 0, 50, (5, 20))
        filtered_df=filtered_df[filtered_df['exceso_velocidad_kmh']>exceso_velocidad[0]]
        filtered_df=filtered_df[filtered_df['exceso_velocidad_kmh']<exceso_velocidad[1]]
        filtro_inteligente=st.sidebar.checkbox("Filtrado Inteligente de los datos", value=True)
        if filtro_inteligente:

            # Convertir la columna 'fecha_hora' al tipo datetime si aún no lo es
            filtered_df['fecha_hora'] = pd.to_datetime(filtered_df['fecha_hora'])

            # Calcular la diferencia porcentual entre 'velocidad_kmh' y 'Velocidad_Ecu'
            filtered_df['diferencia_porcentual'] = abs((filtered_df['velocidad_kmh'] - filtered_df['Velocidad_Ecu']) / filtered_df['velocidad_kmh'] * 100)

            # Filtrar los registros que cumplen con la condición de diferencia porcentual
            filtered_df = filtered_df[filtered_df['diferencia_porcentual'] <= 20]

            # Definir un periodo de 2 minutos
            periodo = pd.Timedelta(minutes=5)
            filtered_df = filtered_df.drop_duplicates(subset=['fecha_hora', 'patente'])


            # Agrupar por patente y por cada periodo de 2 minutos, luego seleccionar el registro con el valor más alto de 'exceso_velocidad_kmh'
            filtered_df = filtered_df.groupby(['patente', pd.Grouper(key='fecha_hora', freq=periodo)])['exceso_velocidad_kmh'].idxmax().apply(lambda x: filtered_df.loc[x])

        if filtered_df.empty:
            st.error("Sin Transgresiones")

        # Filtrar el DataFrame según las geocercas seleccionadas
        else:
    

            # Primero, asignamos un color único a cada registro en el DataFrame
            filtered_df['color'] = np.arange(len(filtered_df))

            # Ahora, creamos el gráfico de barras apiladas con el DataFrame modificado
            fig = px.bar(filtered_df, x='exceso_velocidad_kmh', y=np.ones(filtered_df.shape[0]), title='Transgresiones de Velocidad Excesos Registrados',
                        labels={'exceso_velocidad_kmh': 'Exceso de Velocidad (km/h)', 'y': 'Cantidad de Registros'},
                        hover_data=['velocidad_kmh', 'geocerca', 'nombre', 'patente', 'fecha_hora','Velocidad_Ecu'],
                        color='color',  # Asignar un color distinto a cada registro
                        color_continuous_scale='Cividis')  # Usar la escala de colores 'Inferno'

            fig.update_layout(barmode='stack')
            fig.update_layout(width=1000, height=500)
            col1, col2=st.columns(2)
            with col1:
                st.plotly_chart(fig, use_container_width=True)

            # Ahora, creamos el gráfico de barras apiladas con el DataFrame modificado
            fig = px.bar(filtered_df, x='velocidad_kmh', y=np.ones(filtered_df.shape[0]), title='Transgresiones de Velocidad Registradas',
                        labels={'velocidad_kmh': ' Velocidad (km/h)', 'y': 'Cantidad de Registros'},
                        hover_data=['velocidad_kmh', 'geocerca', 'nombre', 'patente', 'fecha_hora',],
                        color='color',  # Asignar un color distinto a cada registro
                        color_continuous_scale='Viridis')  # Usar la escala de colores 'Inferno'

            fig.update_layout(barmode='stack')
            fig.update_layout(width=1000, height=500)

            with col2:
                st.plotly_chart(fig, use_container_width=True)



            data = filtered_df[['apellido', 'nombre']].value_counts().reset_index()
            data.columns = ['apellido', 'nombre', 'count']
            data_grafico = data[data['apellido'] != 'Sin registro']
            data_grafico = data_grafico[data_grafico['apellido'] != 'Sin Registro']
            col1, col2=st.columns(2)
            with col1:
                st.markdown("**Transgresiones de Velocidad por Operador**")

                pie = (
                    Pie()
                    .add("", [list(z) for z in zip(data_grafico['apellido'], data_grafico['count'])], radius=["40%", "75%"])
                    .set_global_opts(
                        legend_opts=opts.LegendOpts(is_show=False),
                        graphic_opts=[
                            opts.GraphicText(
                                graphic_item=opts.GraphicItem(
                                    left="center",
                                    top="center",
                                    z=1
                                ),
                                graphic_textstyle_opts=opts.GraphicTextStyleOpts(
                                    text=f"{len(data_grafico)} Infractores",
                                    font="bold 17px Microsoft YaHei",
                                    graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(fill="#333")
                                )
                            )
                        ]
                    )
                    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}"),
                                    tooltip_opts=opts.TooltipOpts(formatter="{b}: {c} ({d}%)"))
                )
                st_pyecharts(pie)
            with col2:


                data = filtered_df[['geocerca']].value_counts().reset_index()
                st.markdown("**Transgresiones por Geocerca**")
                pie = (
                    Pie()
                    .add("", [list(z) for z in zip(data['geocerca'], data['count'])], radius=["40%", "75%"])
                    .set_global_opts(
                        legend_opts=opts.LegendOpts(is_show=False), # Oculta la leyenda de colores
                        graphic_opts=[
                            opts.GraphicText(
                                graphic_item=opts.GraphicItem(
                                    left="center",
                                    top="center",
                                    z=1
                                ),
                                graphic_textstyle_opts=opts.GraphicTextStyleOpts(
                                    text=f"{len(data)} Geocercas",
                                    font="bold 17px Microsoft YaHei",
                                    graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(fill="#333")
                                )
                            )
                        ]
                    )
                    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}"), # Muestra solo el apellido
                                    tooltip_opts=opts.TooltipOpts(formatter="{b}: {c} ({d}%)")) # Muestra el número y el porcentaje solo al pasar el mouse por encima
                )
                # Mostrar el gráfico en Streamlit
                st_pyecharts(pie)

            st.write(filtered_df)

if funcion=="En Desarrollo 2":
    base_id='appUIz9SCHdcZDk1T'
    table_id='tblzZLoGJAP3cDwhj'
    personal_access_token='patii9YeJWbaL2hRu.c4f92462f3b6cb1b0f43ba1e6194ab9266209c836167eca08688594167590d0d'
    df = pd.read_csv('https://raw.githubusercontent.com/MatiasCatalanR/Talabre/main/Copia%20de%20Prog_Rellenos.csv', sep=';', index_col=False)
    df


    st.title("🧮 Programación Rellenos")



    data_area = df.groupby('ÁREA')['TOTAL (M3)'].sum()
    st.markdown("**Total m³ por Área**")

    # Creamos una lista de diccionarios para la opción de datos en la serie
    data_list_area = [{"value": v, "name": n} for n, v in data_area.items()]

    # Calcular la suma total de los valores en data_list_area
    total = sum(item['value'] for item in data_list_area)

    print(f"La suma total de los valores en data_list_area es {total}")

    # Calcular el total
    #total = df['TOTAL (M3)'].sum()
    #total =data_area[data_area.columns[0]].sum()

    # Definir los colores de viridis
    colormap = [
        "#14C8E6", # C 20 M 79 Y 90 K 10
        "#FF6B02", # C 0 M 42 Y 98 K 0
        "#FF165A", # C 0 M 75 Y 89 K 0
        "#00A1C9", # C 80 M 15 Y 30 K 0
        "#BA4A28", # R 186 G 74 B 40
        "#F8A302", # R 248 G 163 B 2
        "#F15A56", # R 241 G 90 B 86
        "#009EB0", # R 0 G 158 B 176
        "#3D4552", # C 78 M 61 Y 46 K 38
        "#7B1E20", # C 43 M 99 Y 90 K 31
        "#C7B299", # C 22 M 27 Y 39 K 6
        "#004E52", # C 87 M 35 Y 46 K 52
        "#1D71B8", # C 85 M 50 Y 0 K 0
        "#764A4A", # C 50 M 70 Y 60 K 31
        "#BB5726", # bb5726
        "#F4A700", # f4a700
        "#E96C28", # e96c28
        "#209EB0", # 209eb0
        "#76151F", # 76151f
        "#374752", # 374752
        "#C8B499", # c8b499
        "#004C4E", # 004c4e
        "#6E4D48"  # 6e4d48
    ]


    import matplotlib.colors
    import numpy as np

    # Genera una paleta de colores "cividis" para 19 colores
    colormap = plt.cm.cividis(np.linspace(0, 1, 19))

    # Convierte los colores a formato hexadecimal
    colormap = [matplotlib.colors.rgb2hex(color) for color in colormap]

    print(colormap)



    # Asignar un color único a cada área
    for i, item in enumerate(data_list_area):
        item['itemStyle'] = {"color": colormap[i % len(colormap)]}

    # Ordenar data_list_area en orden descendente según el valor
    data_list_area.sort(key=lambda x: x['value'], reverse=True)
    
    options = {
        "tooltip": {"trigger": "item"},
        "legend": {"show": False},
        "series": [
            {
                "name": "ÁREA",
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": False,
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": "#fff",
                    "borderWidth": 2,
                },
                "label": {
                    "show": True,
                    "position": "outside",
                    "formatter": "{b}: {c} ({d}%)"
                },
                "emphasis": {
                    "label": {"show": True, "fontSize": "20", "fontWeight": "bold"}
                },
                "labelLine": {"show": True},
                "data": data_list_area,
            },
            {
                "name": 'Total',
                "type": 'gauge',
                "startAngle": 90,
                "endAngle": -269.9999,
                "radius": '50%',
                "pointer": {"show": False},
                "detail": {"show": True, "fontSize": 20, "offsetCenter": [0, '40%']},
                "data": [{'value': total, 'name': 'Total M3'}],
                "title": {"show": False},
                "axisLine": {"show": False},
                "splitLine": {"show": False},
                "axisTick": {"show": False},
                "axisLabel": {"show": False},
                "anchor": {"show": False}
            }
        ],
        
    }



    st_echarts(options=options, height="400px")

    st.write(data_area)




    # Acceder a los nombres de las columnas del 15 al 42
    columnas = df.columns[21:43]  # Python usa indexación basada en 0, por lo que debes restar 1 a los índices

    # Lista de columnas de fechas 
    fechas=columnas

    # Crear un nuevo dataframe con las columnas requeridas
    new_df = pd.melt(df, id_vars=['ÁREA'], value_vars=fechas, var_name='Fecha', value_name='Metros Cúbicos')

    # Eliminar filas con valores nulos en la columna 'Metros Cúbicos'
    new_df = new_df.dropna(subset=['Metros Cúbicos'])
    # Convertir los datos a tipo entero
    new_df['Metros Cúbicos'] = new_df['Metros Cúbicos'].astype(int)

    # Calcular la Metros Cúbicos para cada 'ÁREA' y 'Fecha'
    new_df = new_df.groupby(['ÁREA', 'Fecha'])['Metros Cúbicos'].sum().reset_index()
    
    # Calcular el total para cada 'Fecha'
    total_df = new_df.groupby('Fecha')['Metros Cúbicos'].sum().reset_index()
    
    #new_df=new_df.replace('-', '', inplace=True)


    #total_df.columns = ['Fecha', 'Total']
    

    # Unir el dataframe original con los totales
    #new_df = pd.merge(new_df, total_df, on='Fecha')

    # Ordena el DataFrame por 'Metros Cúbicos VP' en orden descendente
    new_df = new_df.sort_values('ÁREA')
    # Crea un diccionario que mapea cada 'ÁREA' a un color
    color_dict = {area: colormap[i % len(colormap)] for i, area in enumerate(new_df['ÁREA'].unique())}
    st.markdown("**Metros Cúbicos Mensuales a Transportar por Área**")
    # Convertir la columna 'Fecha' a tipo datetime
    new_df['Fecha'] = pd.to_datetime(new_df['Fecha'], format='%d-%m-%Y')

    # Obtener el año y el mes de la fecha
    new_df['Año_Mes'] = new_df['Fecha'].dt.strftime('%Y-%m')

    # Crear el gráfico de barras apiladas
    chart = alt.Chart(new_df).mark_bar().encode(
        x=alt.X('Año_Mes:O', sort='ascending'),  # Ordenar las fechas de manera ascendente
        y='Metros Cúbicos:Q',
        color=alt.Color('ÁREA:N', scale=alt.Scale(scheme="cividis")),
        tooltip=[
            'Fecha:N', 
            alt.Tooltip('ÁREA:N'),
            alt.Tooltip('Metros Cúbicos_x:Q', format=',d'),
            alt.Tooltip('Total:Q', format=',d')
        ]
    ).properties(
        width=800,
        height=500
    )

    chart



if funcion== "En Desarrollo 1":

    st.title("📚 MDG 2024 (En Desarrollo)")
        # Crear DataFrame con los datos de los muros y las cantidades diarias depositadas
    import plotly.express as px
    import pandas as pd
    from datetime import timedelta
    import streamlit as st
    fecha_actual = np.datetime64(datetime.now().date())
    
    # Crear DataFrame con los datos de los muros y las cantidades diarias depositadas
    muros = ['MURO OESTE Y NO 1', 'MURO SUR', 'PRETIL 2 NS', 'PRETIL PPAL EBMN', 'MURO NORTE']
    
    # Crear rango de fechas desde el 5 de junio hasta el 31 de diciembre
    fechas = pd.date_range(start=fecha_actual, end='2024-12-31')


    # Crear un DataFrame con una fila para cada combinación de fecha y muro
    df_p = pd.DataFrame([(fecha, muro, 3000) for fecha in fechas for muro in muros], columns=['Fecha', 'Muro', 'Cantidad Diaria'])

    # Ajustar la cantidad diaria para los días con cambio de turno
    ultimo_cambio = pd.Timestamp('2024-05-31')
    while ultimo_cambio < df_p['Fecha'].max():
        ultimo_cambio += timedelta(days=10)
        df_p.loc[df_p['Fecha'] == ultimo_cambio, 'Cantidad Diaria'] = 2100

    # Definir los colores personalizados
    colores = ["#f4a700", "#374752", "#c8b499", "#bb5726", "#76151f"]
    # Crear gráfico de barras apiladas utilizando Plotly Express
    fig = px.bar(df_p, x='Fecha', y='Cantidad Diaria', color='Muro', barmode='stack', color_discrete_sequence=colores)
    fig.update_layout(title='Proyección de Depósito Diario en Muros', xaxis_title='Fecha', yaxis_title='Cantidad (m³)')
    fig.update_layout(width=900, height=500)
    #st.plotly_chart(fig)

    # Calcular la fecha del próximo cambio de turno
    siguiente_cambio_turno = ultimo_cambio + timedelta(days=10)
#### parte analisis historico
    base_id='appOSiiFDWPc1n9hk'
    table_id='tblEKC7vFnnSxQ8JB'
    personal_access_token='patxqLGT8vqeIOi0S.0a930afb2c9db20d8ccfef398bbaafa7f3149a29f5c536e28aeae29f37fac516'
    
    def create_headers():
        headers = {
        'Authorization': 'Bearer ' + str(personal_access_token),
        'Content-Type': 'application/json',
        }
        return headers

    base_table_api_url = 'https://api.airtable.com/v0/{}/{}'.format(base_id, table_id)
    headers=create_headers()
    response = requests.get(base_table_api_url, headers=headers)
    if response.status_code == 200:
        # Convierte la respuesta en formato JSON a un diccionario
        data = response.json()
        
        # Extrae los registros de la respuesta
        records = data['records']
        
        # Crea una lista vacía para almacenar los datos de cada registro
        rows = []
        
        # Recorre cada registro y extrae los datos necesarios
        for record in records:
            fields = record['fields']
            rows.append(fields)

        # Crea el DataFrame utilizando la lista de registros
        df = pd.DataFrame(rows)
        


        # Define el orden de las columnas que deseas mantener
        column_order = ['SECCION', 'Total/Fecha', '11/25/23', '11/26/23', '11/27/23', '11/28/23', '11/29/23', '11/30/23', '12/1/23', '12/2/23', '12/3/23', '12/4/23', '12/5/23', '12/6/23', '12/7/23', '12/8/23', '12/9/23', '12/10/23', '12/11/23', '12/12/23', '12/13/23', '12/14/23', '12/15/23', '12/16/23', '12/17/23', '12/18/23', '12/19/23', '12/20/23', '12/21/23', '12/22/23', '12/23/23', '12/24/23', '12/25/23', '12/26/23', '12/27/23', '12/28/23', '12/29/23', '12/30/23', '12/31/23', '1/1/24', '1/2/24', '1/3/24', '1/4/24', '1/5/24', '1/6/24', '1/7/24', '1/8/24', '1/9/24', '1/10/24', '1/11/24', '1/12/24', '1/13/24', '1/14/24', '1/15/24', '1/16/24', '1/17/24', '1/18/24', '1/19/24', '1/20/24', '1/21/24', '1/22/24', '1/23/24', '1/24/24', '1/25/24', '1/26/24', '1/27/24', '1/28/24', '1/29/24', '1/30/24', '1/31/24', '2/1/24', '2/2/24', '2/3/24', '2/4/24', '2/5/24', '2/6/24', '2/7/24', '2/8/24', '2/9/24', '2/10/24', '2/11/24', '2/12/24', '2/13/24', '2/14/24', '2/15/24', '2/16/24', '2/17/24', '2/18/24', '2/19/24', '2/20/24', '2/21/24', '2/22/24', '2/23/24', '2/24/24', '2/25/24', '2/26/24', '2/27/24', '2/28/24','2/29/24' ,'3/1/24', '3/2/24', '3/3/24', '3/4/24', '3/5/24', '3/6/24', '3/7/24', '3/8/24', '3/9/24', '3/10/24', '3/11/24', '3/12/24', '3/13/24', '3/14/24', '3/15/24', '3/16/24', '3/17/24', '3/18/24', '3/19/24', '3/20/24', '3/21/24', '3/22/24', '3/23/24', '3/24/24', '3/25/24', '3/26/24', '3/27/24', '3/28/24', '3/29/24', '3/30/24', '3/31/24', '4/1/24', '4/2/24', '4/3/24', '4/4/24', '4/5/24', '4/6/24', '4/7/24', '4/8/24', '4/9/24', '4/10/24', '4/11/24', '4/12/24', '4/13/24', '4/14/24','4/15/24', '4/16/24', '4/17/24', '4/18/24', '4/19/24', '4/20/24', '4/21/24', '4/22/24', '4/23/24', '4/24/24', '4/25/24', '4/26/24', '4/27/24', '4/28/24', '4/29/24', '4/30/24', '5/1/24', '5/2/24', '5/3/24', '5/4/24', '5/5/24', '5/6/24', '5/7/24', '5/8/24', '5/9/24', '5/10/24', '5/11/24', '5/12/24', '5/13/24', '5/14/24', '5/15/24', '5/16/24', '5/17/24', '5/18/24', '5/19/24', '5/20/24', '5/21/24', '5/22/24', '5/23/24', '5/24/24','5/25/24',"5/26/24", "5/27/24", "5/28/24", "5/29/24", "5/30/24", "5/31/24", "6/1/24", "6/2/24", "6/3/24", "6/4/24", "6/5/24", "6/6/24", "6/7/24", "6/8/24", "6/9/24", "6/10/24", "6/11/24", "6/12/24", "6/13/24", "6/14/24", "6/15/24", "6/16/24", "6/17/24", "6/18/24", "6/19/24", "6/20/24", "6/21/24", "6/22/24", "6/23/24", "6/24/24", "6/25/24", "6/26/24", "6/27/24", "6/28/24", "6/29/24", "6/30/24", "7/1/24", "7/2/24", "7/3/24", "7/4/24", "7/5/24", "7/6/24", "7/7/24", "7/8/24", "7/9/24", "7/10/24", "7/11/24", "7/12/24", "7/13/24", "7/14/24", "7/15/24", "7/16/24", "7/17/24", "7/18/24", "7/19/24", "7/20/24", "7/21/24", "7/22/24", "7/23/24", "7/24/24", "7/25/24", "7/26/24", "7/27/24", "7/28/24", "7/29/24", "7/30/24", "7/31/24", "8/1/24", "8/2/24", "8/3/24", "8/4/24", "8/5/24", "8/6/24", "8/7/24", "8/8/24", "8/9/24", "8/10/24", "8/11/24", "8/12/24", "8/13/24", "8/14/24", "8/15/24", "8/16/24", "8/17/24", "8/18/24", "8/19/24", "8/20/24", "8/21/24", "8/22/24", "8/23/24", "8/24/24", "8/25/24", "8/26/24", "8/27/24", "8/28/24", "8/29/24", "8/30/24", "8/31/24", "9/1/24", "9/2/24", "9/3/24", "9/4/24", "9/5/24", "9/6/24", "9/7/24", "9/8/24", "9/9/24", "9/10/24", "9/11/24", "9/12/24", "9/13/24", "9/14/24", "9/15/24", "9/16/24", "9/17/24", "9/18/24", "9/19/24", "9/20/24", "9/21/24", "9/22/24", "9/23/24", "9/24/24", "9/25/24", "9/26/24", "9/27/24", "9/28/24", "9/29/24", "9/30/24", "10/1/24", "10/2/24", "10/3/24", "10/4/24", "10/5/24", "10/6/24", "10/7/24", "10/8/24", "10/9/24", "10/10/24", "10/11/24", "10/12/24", "10/13/24", "10/14/24", "10/15/24", "10/16/24", "10/17/24", "10/18/24", "10/19/24", "10/20/24", "10/21/24", "10/22/24", "10/23/24", "10/24/24", "10/25/24", "10/26/24", "10/27/24", "10/28/24", "10/29/24", "10/30/24", "10/31/24", "11/1/24", "11/2/24", "11/3/24", "11/4/24", "11/5/24", "11/6/24", "11/7/24", "11/8/24", "11/9/24", "11/10/24", "11/11/24", "11/12/24", "11/13/24", "11/14/24", "11/15/24", "11/16/24", "11/17/24", "11/18/24", "11/19/24", "11/20/24", "11/21/24", "11/22/24", "11/23/24", "11/24/24", "11/25/24", "11/26/24", "11/27/24", "11/28/24", "11/29/24", "11/30/24", "12/1/24", "12/2/24", "12/3/24", "12/4/24", "12/5/24", "12/6/24", "12/7/24", "12/8/24", "12/9/24", "12/10/24", "12/11/24", "12/12/24", "12/13/24", "12/14/24", "12/15/24", "12/16/24", "12/17/24", "12/18/24", "12/19/24", "12/20/24", "12/21/24", "12/22/24", "12/23/24", "12/24/24", "12/25/24", "12/26/24", "12/27/24", "12/28/24", "12/29/24", "12/30/24", "12/31/24", "1/1/25", "1/2/25", "1/3/25", "1/4/25", "1/5/25", "1/6/25", "1/7/25", "1/8/25", "1/9/25", "1/10/25", "1/11/25", "1/12/25", "1/13/25", "1/14/25", "1/15/25", "1/16/25", "1/17/25", "1/18/25", "1/19/25", "1/20/25", "1/21/25", "1/22/25", "1/23/25", "1/24/25", "1/25/25", "1/26/25", "1/27/25", "1/28/25", "1/29/25", "1/30/25", "1/31/25", "2/1/25", "2/2/25", "2/3/25", "2/4/25", "2/5/25", "2/6/25", "2/7/25", "2/8/25", "2/9/25", "2/10/25", "2/11/25", "2/12/25", "2/13/25", "2/14/25", "2/15/25", "2/16/25", "2/17/25", "2/18/25", "2/19/25", "2/20/25", "2/21/25", "2/22/25", "2/23/25", "2/24/25", "2/25/25", "2/26/25", "2/27/25", "2/28/25", "3/1/25", "3/2/25", "3/3/25", "3/4/25", "3/5/25", "3/6/25", "3/7/25", "3/8/25", "3/9/25", "3/10/25", "3/11/25", "3/12/25", "3/13/25", "3/14/25", "3/15/25", "3/16/25", "3/17/25", "3/18/25", "3/19/25", "3/20/25", "3/21/25", "3/22/25", "3/23/25", "3/24/25", "3/25/25", "3/26/25", "3/27/25", "3/28/25", "3/29/25", "3/30/25", "3/31/25", "4/1/25", "4/2/25", "4/3/25", "4/4/25", "4/5/25", "4/6/25", "4/7/25", "4/8/25", "4/9/25", "4/10/25", "4/11/25", "4/12/25", "4/13/25", "4/14/25", "4/15/25", "4/16/25", "4/17/25", "4/18/25", "4/19/25", "4/20/25", "4/21/25", "4/22/25", "4/23/25", "4/24/25", "4/25/25", "4/26/25", "4/27/25", "4/28/25", "4/29/25", "4/30/25", "5/1/25", "5/2/25", "5/3/25", "5/4/25", "5/5/25", "5/6/25", "5/7/25", "5/8/25", "5/9/25", "5/10/25", "5/11/25", "5/12/25", "5/13/25", "5/14/25", "5/15/25", "5/16/25", "5/17/25", "5/18/25", "5/19/25", "5/20/25", "5/21/25", "5/22/25", "5/23/25", "5/24/25", "5/25/25", "5/26/25", "5/27/25", "5/28/25", "5/29/25", "5/30/25", "5/31/25", "6/1/25", "6/2/25", "6/3/25", "6/4/25", "6/5/25", "6/6/25", "6/7/25", "6/8/25", "6/9/25", "6/10/25", "6/11/25", "6/12/25", "6/13/25", "6/14/25", "6/15/25", "6/16/25", "6/17/25", "6/18/25", "6/19/25", "6/20/25", "6/21/25", "6/22/25", "6/23/25", "6/24/25", "6/25/25", "6/26/25", "6/27/25", "6/28/25", "6/29/25", "6/30/25", "7/1/25", "7/2/25", "7/3/25", "7/4/25", "7/5/25", "7/6/25", "7/7/25", "7/8/25", "7/9/25", "7/10/25", "7/11/25", "7/12/25", "7/13/25", "7/14/25", "7/15/25", "7/16/25", "7/17/25", "7/18/25", "7/19/25", "7/20/25", "7/21/25", "7/22/25", "7/23/25", "7/24/25", "7/25/25", "7/26/25", "7/27/25", "7/28/25", "7/29/25", "7/30/25", "7/31/25", "8/1/25", "8/2/25", "8/3/25", "8/4/25", "8/5/25", "8/6/25", "8/7/25", "8/8/25", "8/9/25", "8/10/25", "8/11/25", "8/12/25", "8/13/25", "8/14/25", "8/15/25", "8/16/25", "8/17/25", "8/18/25", "8/19/25", "8/20/25", "8/21/25", "8/22/25", "8/23/25", "8/24/25", "8/25/25", "8/26/25", "8/27/25", "8/28/25", "8/29/25", "8/30/25", "8/31/25", "9/1/25", "9/2/25", "9/3/25", "9/4/25", "9/5/25", "9/6/25", "9/7/25", "9/8/25", "9/9/25", "9/10/25", "9/11/25", "9/12/25", "9/13/25", "9/14/25", "9/15/25", "9/16/25", "9/17/25", "9/18/25", "9/19/25", "9/20/25", "9/21/25", "9/22/25", "9/23/25", "9/24/25", "9/25/25", "9/26/25", "9/27/25", "9/28/25", "9/29/25", "9/30/25", "10/1/25", "10/2/25", "10/3/25", "10/4/25", "10/5/25", "10/6/25", "10/7/25", "10/8/25", "10/9/25", "10/10/25", "10/11/25", "10/12/25", "10/13/25", "10/14/25", "10/15/25", "10/16/25", "10/17/25", "10/18/25", "10/19/25", "10/20/25", "10/21/25", "10/22/25", "10/23/25", "10/24/25", "10/25/25", "10/26/25", "10/27/25", "10/28/25", "10/29/25", "10/30/25", "10/31/25", "11/1/25", "11/2/25", "11/3/25", "11/4/25", "11/5/25", "11/6/25", "11/7/25", "11/8/25", "11/9/25", "11/10/25", "11/11/25", "11/12/25", "11/13/25", "11/14/25", "11/15/25", "11/16/25", "11/17/25", "11/18/25", "11/19/25", "11/20/25", "11/21/25", "11/22/25", "11/23/25", "11/24/25", "11/25/25", "11/26/25", "11/27/25", "11/28/25", "11/29/25", "11/30/25", "12/1/25", "12/2/25", "12/3/25", "12/4/25", "12/5/25", "12/6/25", "12/7/25", "12/8/25", "12/9/25", "12/10/25", "12/11/25", "12/12/25", "12/13/25", "12/14/25", "12/15/25", "12/16/25", "12/17/25", "12/18/25", "12/19/25", "12/20/25", "12/21/25", "12/22/25", "12/23/25", "12/24/25", "12/25/25", "12/26/25", "12/27/25"]

        # Reindexa el DataFrame utilizando el orden de las columnas
        df = df.reindex(columns=column_order)
        
    mapeo = {
        'ZAMPEADO PRETIL PRINCIPAL T3 4.2.1.1.1': 'PRETIL PPAL EBMN',
        'RELLENO PRETIL PRINCIPAL T3 4.2.1.1.2': 'PRETIL PPAL EBMN',
        'RELLENO PRETIL PRINCIPAL T3 4.2.1.1.2 INTERNO': 'PRETIL PPAL EBMN',
        'ZAMPEADO PRETIL CONTORNO (ANILLO) EBMN 5.1.2.3.1': 'PRETIL PPAL EBMN',
        'RELLENO PRETIL CONTORNO (ANILLO) EBMN  5.1.2.3.2': 'PRETIL PPAL EBMN',
        'ZAMPEADO PRETIL PRINCIPAL T1 4.2.1.1.1': 'PRETIL PPAL EBMN',
        'RELLENO PRETIL PRINCIPAL T1 4.2.1.1.2': 'PRETIL PPAL EBMN',
        'RELLENO PLATAFORMAS 1,2,3,4,5 y 6 EBMN  5.1.2.2.2': 'PRETIL PPAL EBMN',
        'ZAMPEADO PLATAFORMAS EBMN 1,2,3,4,5 y 6  5.1.2.2.1': 'PRETIL PPAL EBMN',
        'RELLENO ZAMPEADO CAMINO PRETIL PRINCIPAL T1 4.2.1.1.1': 'CAMINOS ZAMPEADO',
        'BYPASS EBMN (ZAMPEADO)': 'CAMINOS ZAMPEADO',         
        'RELLENO AGUAS ABAJO MURO OESTE 1.4.1.3.1': 'MURO OESTE Y NO 1',
        'RELLENO AGUAS ABAJO MURO NOROESTE 1 1.3.1.2.1': 'MURO OESTE Y NO 1',
        'RELLENO MURO SUR TRAMO 1 1.5.1.3.1': 'MURO SUR',
        'ZAMPEADO PRETIL 2N-S 4.1.1.1.1': 'PRETIL 2 NS',
        'RELLENO PRETIL 2N-S':'PRETIL 2 NS',
        'Total': 'Total'
    }
    
    # Crear la nueva columna 'Unidad de Negocio'
    df['Secciones']=df['SECCION'].map(mapeo)
    
    # Definir la función para aplicar el factor de multiplicación
    def apply_factor(x):
        return (x * 14) / 20

    # Aplicar el factor de multiplicación a las columnas seleccionadas
    fechas_factor=['11/25/23', '11/26/23', '11/27/23', '11/28/23', '11/29/23', '11/30/23', '12/1/23', '12/2/23', '12/3/23', '12/4/23', '12/5/23', '12/6/23', '12/7/23', '12/8/23', '12/9/23', '12/10/23', '12/11/23', '12/12/23', '12/13/23', '12/14/23', '12/15/23', '12/16/23', '12/17/23', '12/18/23', '12/19/23', '12/20/23', '12/21/23', '12/22/23', '12/23/23', '12/24/23', '12/25/23', '12/26/23', '12/27/23', '12/28/23', '12/29/23', '12/30/23', '12/31/23', '1/1/24', '1/2/24', '1/3/24', '1/4/24', '1/5/24', '1/6/24', '1/7/24', '1/8/24', '1/9/24', '1/10/24', '1/11/24', '1/12/24', '1/13/24', '1/14/24', '1/15/24', '1/16/24', '1/17/24', '1/18/24', '1/19/24', '1/20/24', '1/21/24', '1/22/24', '1/23/24', '1/24/24', '1/25/24', '1/26/24', '1/27/24', '1/28/24', '1/29/24', '1/30/24', '1/31/24', '2/1/24', '2/2/24', '2/3/24', '2/4/24', '2/5/24', '2/6/24', '2/7/24', '2/8/24', '2/9/24', '2/10/24', '2/11/24', '2/12/24', '2/13/24']

    df[fechas_factor] = df[fechas_factor].apply(apply_factor)        

    df_limpieza = df.drop("SECCION", axis=1)
    df_limpieza = df_limpieza.groupby("Secciones").sum()
    df_limpieza = df_limpieza.reset_index()

    df_total = df_limpieza.loc[df_limpieza['Secciones'] == 'Total']
    columnas_suma = df_total.columns[2:]
    suma_total = df_total[columnas_suma].sum()
    suma_total_historica=suma_total.sum()



    # Crear un nuevo dataframe con la columna 'secciones'
    df_secciones = pd.DataFrame(df_limpieza['Secciones'])
    

    # Calcula la media móvil de 10 días para cada sección
    df_limpieza_media_movil = df_limpieza.iloc[:, 2:].rolling(window=10, axis=1).mean()


    # Concatenar el nuevo dataframe con 'df_limpieza_mediamovil'
    df_limpieza_media_movil = pd.concat([df_limpieza_media_movil, df_secciones], axis=1)
    #st.write(df_final)
    # Imprime el nuevo dataframe con la media móvil de 10 días

    # Obtener las secciones del DataFrame
    secciones = df_limpieza_media_movil["Secciones"].unique()
    
    # Crear un DataFrame para los valores y las fechas
    data = pd.DataFrame(columns=["Fecha", "Metros Cúbicos Compactados", "Seccion"])
    lista_fecha=[]

    for seccion in secciones:
        # Obtener los valores de la sección
        valores = df_limpieza_media_movil[df_limpieza_media_movil["Secciones"] == seccion].iloc[:, 2:-1]
        
        # Convertir las fechas a formato datetime
        fechas = valores.columns  # Las fechas ya están en el formato correcto
        
        # Crear un DataFrame con las fechas, valores y sección
        temp_df = pd.DataFrame({
            "Fecha": fechas,
            "Metros Cúbicos Compactados": valores.values[0],
            "Seccion": seccion
        })
        ultima_fecha = temp_df[temp_df["Metros Cúbicos Compactados"] != 0]["Fecha"].max()

        # Agregar el DataFrame temporal al DataFrame principal
        data = pd.concat([data, temp_df])
        lista_fecha.append(ultima_fecha)

    lista_fecha = max(lista_fecha)
    
    lista_fecha=pd.to_datetime(lista_fecha)
    
    

    # Convertir la columna de fechas a tipo datetime
    data["Fecha"] = pd.to_datetime(data["Fecha"])
    # Restar un día a las fechas
    fechas_ayer = fecha_actual - np.timedelta64(2                                                                                                                                                                                 , 'D')
    
    data = data[data["Fecha"] <= fechas_ayer]

    # Definir los colores deseados

    # Definir los colores para las rectas
    colores = ["#f4a700", "#374752", "#c8b499", "#bb5726", "#76151f"]

    # Graficar los valores utilizando plotly
    fig = px.line(data, x="Fecha", y="Metros Cúbicos Compactados", color="Seccion", title="Rellenos Compactados Media Móvil 10 días")

    # Modificar el color de las rectas
    for i, color in enumerate(colores):
        fig.data[i].line.color = color

    # Añadir una recta para el valor 3000
    fig.add_trace(go.Scatter(x=data["Fecha"], y=[3000] * len(data), mode="lines", name="Meta Diaria Frentes", line_color="#209eb0"))
    # Añadir una recta para el valor 15000
    fig.add_trace(go.Scatter(x=data["Fecha"], y=[15000] * len(data), mode="lines", name="Meta Diaria Total", line_color="#bb5726"))

    fig.update_layout(width=900, height=500)
    st.plotly_chart(fig, use_container_width=True)
    # Obtener las secciones del DataFrame
    secciones = df_limpieza_media_movil["Secciones"].unique()

    # Crear un DataFrame para los valores y las fechas
    data = pd.DataFrame(columns=["Fecha", "Metros Cúbicos Compactados", "Seccion"])

    # Obtener los valores y las fechas para cada sección
    for seccion in secciones:
        # Obtener los valores de la sección
        valores = df_limpieza_media_movil[df_limpieza_media_movil["Secciones"] == seccion].iloc[:, 2:-1]
        
        # Convertir las fechas a formato datetime
        fechas = valores.columns  # Las fechas ya están en el formato correcto
        
        # Crear un DataFrame con las fechas, valores y sección
        temp_df = pd.DataFrame({
            "Fecha": fechas,
            "Metros Cúbicos Compactados": valores.values[0],
            "Seccion": seccion
        })
        
        # Agregar el DataFrame temporal al DataFrame principal
        data = pd.concat([data, temp_df])

    # Convertir la columna de fechas a tipo datetime
    data["Fecha"] = pd.to_datetime(data["Fecha"])
    data = data[data["Fecha"] <= fechas_ayer]
    data_total = data[data['Seccion'] == 'Total']
    data = data[data['Seccion'] != 'Total']
    

    df_total = df_limpieza[df_limpieza["Secciones"] == "Total"]

    #PRUEBA
    valores = df_total[df_total["Secciones"] == seccion].iloc[:, 2:-1]
    
    # Convertir las fechas a formato datetime
    fechas = valores.columns  # Las fechas ya están en el formato correcto
    
    # Crear un DataFrame con las fechas, valores y sección
    temp_df = pd.DataFrame({
        "Fecha": fechas,
        "Metros Cúbicos Compactados": valores.values[0],
        "Seccion": seccion
    })
    import plotly.graph_objects as go

    # Agregar el DataFrame temporal al DataFrame principal
    data_total = temp_df
    data_total["Fecha"] = pd.to_datetime(data_total["Fecha"])
    data_total = data_total[data_total["Fecha"] <= fechas_ayer]
    #PRUEBA
    fig = px.bar(data, x="Fecha", y="Metros Cúbicos Compactados", color="Seccion", title="Rellenos Compactados Medias Móviles 10 días Apiladas", barmode="stack", color_discrete_sequence=["#f4a700", "#374752", "#c8b499", "#bb5726", "#76151f"])
    # Crear el gráfico de dispersión y asignarle un nombre para la leyenda
    # Crear el trace adicional utilizando go.Scatter() y asignarle un nombre para la leyenda
    scatter_trace = go.Scatter(
        x=data_total["Fecha"],
        y=data_total["Metros Cúbicos Compactados"],
        mode="markers",
        marker=dict(
            symbol="x",  # Cambia la forma de los puntos a cuadrados
            size=5,  # Cambia el tamaño de los puntos
            color="black"  # Cambia el color de los puntos
        ),
        name="Relleno Total Diario"
    )
    # Agregar el trace adicional al gráfico de barras
    fig.add_trace(scatter_trace)

    fig.update_layout(width=900, height=500)




    # Crear DataFrame con los datos de los muros y las cantidades diarias depositadas
    muros = ['MURO OESTE Y NO 1', 'MURO SUR', 'PRETIL 2 NS', 'PRETIL PPAL EBMN', 'CAMINOS ZAMPEADO']

    # Crear rango de fechas desde el 5 de junio hasta el 31 de diciembre
    fechas = pd.date_range(start='2024-06-05', end='2024-12-31')

    # Crear un DataFrame con una fila para cada combinación de fecha y muro
    suma_proyectado=df_p['Cantidad Diaria'].sum()
    style_metric_cards()


    # Ajustar la cantidad diaria para los días con cambio de turno
    ultimo_cambio = pd.Timestamp('2024-05-31')
    while ultimo_cambio < df_p['Fecha'].max():
        ultimo_cambio += timedelta(days=10)
        df_p.loc[df_p['Fecha'] == ultimo_cambio, 'Cantidad Diaria'] = 2100
    df_p['Cantidad Diaria'] = df_p['Cantidad Diaria'].rolling(window=10).mean()

    # Definir los colores personalizados
    colores = ["#f4a700", "#374752", "#c8b499", "#bb5726", "#76151f"]
    df_graf = df_p.rename(columns={'Muro': 'Seccion'})
    df_graf = df_graf.rename(columns={'Cantidad Diaria': 'Metros Cúbicos Compactados'})
    df_graf = pd.concat([df_graf, data])
    # Crear gráfico de barras apiladas utilizando Plotly Express
    fig1 = px.bar(df_p, x='Fecha', y='Cantidad Diaria', color='Muro', barmode='stack', color_discrete_sequence=colores)
    fig1.update_layout(title='Depósito Diario en Muros Actual-Requerido', xaxis_title='Fecha', yaxis_title='Media Móvil 10 días(m³)')
    
    # Crear el gráfico de dispersión y asignarle un nombre para la leyenda
    scatter_trace = go.Scatter(
        x=data_total["Fecha"],
        y=data_total["Metros Cúbicos Compactados"],
        mode="markers",
        marker=dict(
            symbol="x",  # Cambia la forma de los puntos a cuadrados
            size=5,  # Cambia el tamaño de los puntos
            color="black"  # Cambia el color de los puntos
        ),
        name="Relleno Total Diario"
    )

    # Crear el segundo gráfico de barras
    fig2 = px.bar(data, x="Fecha", y="Metros Cúbicos Compactados", color="Seccion", title="Rellenos Compactados Medias Móviles 10 días Apiladas", barmode="stack", color_discrete_sequence=colores)

    # Agregar los trazos del segundo gráfico y el gráfico de dispersión al primer gráfico de barras
    for trace in fig2.data:
        fig1.add_trace(trace)
    fig1.add_trace(scatter_trace)

    fig1.update_layout(width=900, height=500)
    st.plotly_chart(fig1, use_container_width=True)
    col1, col2,col3=st.columns(3)
    suma_actual=int(suma_total_historica)
    suma_diciembre=int(suma_actual)+int(suma_proyectado)

    col1.metric(label="Esperado a diciembre",value=format(suma_diciembre,',').replace(',', '.'),delta=str(round(suma_diciembre/2600000*100,1))+"% del MDG")
    col2.metric(label="Acumulado Histórico",value=format(suma_actual,',').replace(',', '.'),delta=str(round(suma_actual/(suma_diciembre)*100,1))+"% de Cumplimiento")
    col3.metric(label="Total por Depositar",value=format((suma_diciembre-suma_actual),',').replace(',', '.'),delta=str(-round(((suma_diciembre-suma_actual)/suma_diciembre)*100,1))+"% Restante")

### parte analisis historico





        

    st.sidebar.title('Cargar archivo')
    uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV o XLSX", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        #st.write(df)

        data_area = df.groupby('ÁREA')['TOTAL (M3)'].sum()

    




        # Acceder a los nombres de las columnas del 15 al 42
        columnas = df.columns[21:30]  # Python usa indexación basada en 0, por lo que debes restar 1 a los índices

        # Lista de columnas de fechas 
        fechas=columnas

        # Crear un nuevo dataframe con las columnas requeridas
        new_df = pd.melt(df, id_vars=['ÁREA'], value_vars=fechas, var_name='Fecha', value_name='Metros Cúbicos')

        # Convertir la columna 'Fecha' a formato de texto y extraer el mes y el año
        new_df['Fecha'] = pd.to_datetime(new_df['Fecha']).dt.strftime('%Y-%m')

        # Calcular la Metros Cúbicos para cada 'ÁREA' y 'Fecha'
        new_df = new_df.groupby(['ÁREA', 'Fecha'])['Metros Cúbicos'].sum().reset_index()

        # Calcular el total para cada 'Fecha'
        total_df = new_df.groupby('Fecha')['Metros Cúbicos'].sum().reset_index()
        total_df.columns = ['Fecha', 'Total']

        # Unir el dataframe original con los totales
        new_df = pd.merge(new_df, total_df, on='Fecha')

        acciona_diciembre=new_df['Metros Cúbicos'].sum()
        st.write("Exigido a Acciona al 31 Diciembre:",acciona_diciembre)
        exigido_vp_acc=(2600000/acciona_diciembre)*100
        new_df['Metros Cúbicos VP']=new_df['Metros Cúbicos']*(exigido_vp_acc/100)
        # Convierte las columnas a enteros
        new_df['Metros Cúbicos VP'] = new_df['Metros Cúbicos VP'].round().astype(int)


        new_df['Total VP']=new_df['Total']*(exigido_vp_acc/100)
        # Redondea los valores al entero más cercano y luego los convierte a enteros
        new_df['Total VP'] = new_df['Total VP'].round().astype(int)

        st.write("100% Cumplimiento",int(new_df['Metros Cúbicos VP'].sum()))
        st.write("Real Requerido VP",str(round(exigido_vp_acc,2))+"%")
        colormap = [
            "#14C8E6", # C 20 M 79 Y 90 K 10
            "#FF6B02", # C 0 M 42 Y 98 K 0
            "#FF165A", # C 0 M 75 Y 89 K 0
            "#00A1C9", # C 80 M 15 Y 30 K 0
            "#BA4A28", # R 186 G 74 B 40
            "#F8A302", # R 248 G 163 B 2
            "#F15A56", # R 241 G 90 B 86
            "#009EB0", # R 0 G 158 B 176
            "#3D4552", # C 78 M 61 Y 46 K 38
            "#7B1E20", # C 43 M 99 Y 90 K 31
            "#C7B299", # C 22 M 27 Y 39 K 6
            "#004E52", # C 87 M 35 Y 46 K 52
            "#1D71B8", # C 85 M 50 Y 0 K 0
            "#764A4A", # C 50 M 70 Y 60 K 31
            "#BB5726", # bb5726
            "#F4A700", # f4a700
            "#E96C28", # e96c28
            "#209EB0", # 209eb0
            "#76151F", # 76151f
            "#374752", # 374752
            "#C8B499", # c8b499
            "#004C4E", # 004c4e
            "#6E4D48"  # 6e4d48
        ]
        
        # Ordena el DataFrame por 'Metros Cúbicos VP' en orden descendente
        new_df = new_df.sort_values('ÁREA')

        # Crea un diccionario que mapea cada 'ÁREA' a un color
        color_dict = {area: colormap[i % len(colormap)] for i, area in enumerate(new_df['ÁREA'].unique())}
        st.markdown("**Total m³ requeridos por Área**")
        # Crea el gráfico de barras apiladas
        chart = alt.Chart(new_df).mark_bar().encode(
            x='Fecha:N',
            y='Metros Cúbicos VP:Q',
            color=alt.Color('ÁREA:N', scale=alt.Scale(domain=list(color_dict.keys()), range=list(color_dict.values()))),
            tooltip=[
                'Fecha:N', 
                alt.Tooltip('ÁREA:N'),
                alt.Tooltip('Metros Cúbicos VP:Q', format=',d'),
                alt.Tooltip('Total VP:Q', format=',d')
            ],
            order=alt.Order(
            'Fecha:N',
            sort='ascending'
            )
        ).properties(
            width=800,
            height=500
        )



 

        # Agrupa el DataFrame por la columna 'ÁREA' y suma los 'Metros Cúbicos VP'
        grouped_df = new_df.groupby('ÁREA')['Metros Cúbicos VP'].sum()

        # Convierte el objeto GroupBy resultante en un DataFrame
        area_df = pd.DataFrame(grouped_df).reset_index()


        
        # Creamos una lista de diccionarios para la opción de datos en la serie
        data_list_area = [{"value": v, "name": n} for n, v in area_df.set_index('ÁREA')['Metros Cúbicos VP'].to_dict().items()]

        # Calcular la suma total de los valores en data_list_area
        total = sum(item['value'] for item in data_list_area)

        print(f"La suma total de los valores en data_list_area es {total}")

        # Definir los colores de viridis

        # Asignar un color único a cada área
        for i, item in enumerate(data_list_area):
            item['itemStyle'] = {"color": colormap[i % len(colormap)]}

        # Ordenar data_list_area en orden descendente según el valor
        data_list_area.sort(key=lambda x: x['value'], reverse=True)

        options = {
            "tooltip": {"trigger": "item"},
            "legend": {"show": False},
            "series": [
                {
                    "name": "ÁREA",
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": False,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    },
                    "label": {
                        "show": True,
                        "position": "outside",
                        "formatter": "{b}: {c} ({d}%)"
                    },
                    "emphasis": {
                        "label": {"show": True, "fontSize": "20", "fontWeight": "bold"}
                    },
                    "labelLine": {"show": True},
                    "data": data_list_area,
                },
                {
                    "name": 'Total',
                    "type": 'gauge',
                    "startAngle": 90,
                    "endAngle": -269.9999,
                    "radius": '50%',
                    "pointer": {"show": False},
                    "detail": {"show": True, "fontSize": 20, "offsetCenter": [0, '40%']},
                    "data": [{'value': total, 'name': 'Total M3'}],
                    "title": {"show": False},
                    "axisLine": {"show": False},
                    "splitLine": {"show": False},
                    "axisTick": {"show": False},
                    "axisLabel": {"show": False},
                    "anchor": {"show": False}
                }
            ],
        }

        st_echarts(options=options, height="400px")
        st.write(area_df)
        
        st.markdown("**Metros Cúbicos Mensuales Requeridos a Transportar por Área**")

        chart
        st.write(new_df)
if funcion=="Pórtico":
    st.title("🥅 Análisis Pórtico")
    #st.header("Este Análisis contempla el estudio desde el  2024-07-24 al 2024-07-31")



    st.sidebar.title('Cargar archivo')
    uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV o XLSX", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        # Leer el archivo Excel
        df = pd.read_excel(uploaded_file)
    #df=pd.read_csv('https://raw.githubusercontent.com/MatiasCatalanR/Talabre/main/BD%20PORTICO%2030-07.csv', sep=';')
        df
    # Crear el diccionario de mapeo
        mapping = {
            'P27': 'Volvo', 'F92': 'Volvo', 'S79': 'Volvo', 'F31': 'Volvo', 'W58': 'Volvo', 'F33': 'Volvo', 'X32': 'Volvo', 'F94': 'Volvo', 'F93': 'Volvo', 'F51': 'Volvo', 'W59': 'Volvo', 'K46': 'Volvo', 'K47': 'Volvo', 'Z92': 'Volvo', 'L13': 'Volvo', 'V45': 'Volvo', 'F28': 'Volvo', 'S56': 'Volvo', 'K28': 'Volvo', 'H92': 'Volvo', 'B34': 'Volvo', 'K29': 'Volvo', 'F29': 'Volvo', 'V43': 'Volvo', 'K48': 'Volvo',
            'T53': 'Mercedes', 'S36': 'Mercedes', 'F88': 'Mercedes', 'T54': 'Mercedes', 'W10': 'Mercedes', 'H32': 'Mercedes', 'T51': 'Mercedes', 'R84': 'Mercedes', 'T43': 'Mercedes', 'V82': 'Mercedes', 'X15': 'Mercedes', 'Z82': 'Mercedes', 'W78': 'Mercedes', 'J63': 'Mercedes', 'L80': 'Mercedes', 'W12': 'Mercedes', 'L31': 'Mercedes', 'L78': 'Mercedes', 'L67': 'Mercedes', 'L79': 'Mercedes', 'J54': 'Mercedes', 'L77': 'Mercedes', 'L75': 'Mercedes', 'H30': 'Mercedes', 'TT43': 'Mercedes'
        }

        # Crear la nueva columna "Modelo" usando el diccionario de mapeo
        df['Modelo'] = df['Camion ID'].map(mapping)
        df
        df=df[df["Total (m3)"]>0]
        df=df[df["Total (m3)"]>13]

        df_mercedes=df[df["Modelo"]=="Mercedes"]
        df_volvo=df[df["Modelo"]=="Volvo"]


        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        sns.histplot(df["Total (m3)"], color='#76151f')
        plt.xticks(range(int(min(df["Total (m3)"])), int(max(df["Total (m3)"])) + 1))
        plt.text(df["Total (m3)"].mean(), -55, f"Promedio: {df['Total (m3)'].mean():.2f}", ha='center')
        plt.tight_layout()
        plt.title("Histograma Carga Total Camiones (m3)")

        plt.subplot(1, 3, 2)
        sns.histplot(df_mercedes["Total (m3)"], color='#374752')
        plt.xticks(range(int(min(df_mercedes["Total (m3)"])), int(max(df_mercedes["Total (m3)"])) + 1))
        plt.text(df_mercedes["Total (m3)"].mean(), -29, f"Promedio: {df_mercedes['Total (m3)'].mean():.2f}", ha='center')
        plt.tight_layout()
        plt.title("Histograma Carga Camión Mercedes (m3)")

        plt.subplot(1, 3, 3)
        sns.histplot(df_volvo["Total (m3)"], color='#c8b499')
        plt.xticks(range(int(min(df_volvo["Total (m3)"])), int(max(df_volvo["Total (m3)"])) + 1))
        plt.text(df_volvo["Total (m3)"].mean(), -26, f"Promedio: {df_volvo['Total (m3)'].mean():.2f}", ha='center')
        plt.tight_layout()
        plt.title("Histograma Carga Camión Volvo (m3)")

        st.pyplot(plt)




        # Convertir la columna 'Hora' a datetime y extraer la fecha
        df["Hora"] = pd.to_datetime(df["Hora"])
        df["Fecha"] = df["Hora"].dt.date

        # Calcular el promedio diario
        promedio_diario = df.groupby("Fecha")["Total (m3)"].mean().round(1)

        # Crear el gráfico de líneas para el promedio diario
        fig = px.line(promedio_diario, x=promedio_diario.index, y=promedio_diario.values, title="Promedio Diario de m³ Transportados por Viaje y Fecha")
        fig.update_layout(width=900, height=600)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', nticks=10)

        fig.update_xaxes(type='date', tickformat='%Y-%m-%d', dtick="D1")  # Configurar el formato del eje x y el intervalo de ticks
        fig.update_yaxes(title_text='Metros Cúbicos')  # Configurar el título del eje y
        fig.update_traces(line_color='#76151f')
        fig.update_layout(
            xaxis_title_font=dict(size=20),  # Tamaño del título del eje x
            yaxis_title_font=dict(size=20),  # Tamaño del título del eje y
            xaxis=dict(tickfont=dict(size=18)),  # Tamaño de los valores del eje x
            yaxis=dict(tickfont=dict(size=18))   # Tamaño de los valores del eje y
        )
        fig.update_traces(line=dict(width=4))  # Ajusta el valor de 'width' según sea necesario


        col1, col2=st.columns(2)

        # Mostrar el gráfico
        #with col1:
        st.plotly_chart(fig, use_container_width=True)

        # Calcular la suma diaria
        suma_diaria = df.groupby("Fecha")["Total (m3)"].sum()

        # Crear el gráfico de líneas para la suma diaria
        fig2 = px.line(suma_diaria, x=suma_diaria.index, y=suma_diaria.values, title="Total de Metros Cúbicos Medidos en Pórtico por Fecha")
        fig2.update_layout(width=900, height=500)
        fig2.update_xaxes(type='date', tickformat='%Y-%m-%d', dtick="D1")  # Configurar el formato del eje x y el intervalo de ticks
        fig2.update_yaxes(title_text='Total (m3)')  # Configurar el título del eje y
        fig2.update_layout(
            xaxis_title_font=dict(size=20),  # Tamaño del título del eje x
            yaxis_title_font=dict(size=20),  # Tamaño del título del eje y
            xaxis=dict(tickfont=dict(size=18)),  # Tamaño de los valores del eje x
            yaxis=dict(tickfont=dict(size=18))   # Tamaño de los valores del eje y
        )
        fig2.update_traces(line=dict(width=4))
        fig2.update_traces(line_color='#374752')

        st.plotly_chart(fig2, use_container_width=True)
        # Agrupar el DataFrame por la columna 'Fecha' y contar la cantidad de entradas
        cantidad_ciclos = df.groupby("Fecha")["Total (m3)"].count()

        # Crear el gráfico de líneas
        fig2 = px.line(cantidad_ciclos, x=cantidad_ciclos.index, y=cantidad_ciclos.values, title="Ciclos por Fecha Pórtico")
        fig2.update_layout(width=900, height=500)
        fig2.update_xaxes(type='date', tickformat='%Y-%m-%d', dtick="D1")  # Configurar el formato del eje x y el intervalo de ticks
        fig2.update_yaxes(title_text='Cantidad de Ciclos')  # Configurar el título del eje y
        fig2.update_traces(line_color='#374752')
        fig2.update_layout(
            xaxis_title_font=dict(size=20),  # Tamaño del título del eje x
            yaxis_title_font=dict(size=20),  # Tamaño del título del eje y
            xaxis=dict(tickfont=dict(size=18)),  # Tamaño de los valores del eje x
            yaxis=dict(tickfont=dict(size=18))   # Tamaño de los valores del eje y
        )
        fig2.update_traces(line=dict(width=4)) 

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig2, use_container_width=True)
        # Crear el gráfico de barras para la suma diaria
        # Datos proporcionados
        suma_diaria
        data = {
            'Fecha': ['2024-08-14', '2024-08-15', '2024-08-16', '2024-08-17', '2024-08-18', '2024-08-19'],
            'Total (m3)': [9905.9, 11757.7, 10425.6, 12769.4, 11586.8, 5338.3]
        }

        # Crear el DataFrame
        suma_diaria = pd.DataFrame(data)

        # Convertir la columna 'Fecha' a tipo datetime y establecerla como índice
        suma_diaria['Fecha'] = pd.to_datetime(suma_diaria['Fecha'])
        suma_diaria.set_index('Fecha', inplace=True)

        # Crear el gráfico de barras
        fig2 = px.bar(suma_diaria, x=suma_diaria.index, y='Total (m3)', title="Total de Metros Cúbicos Medidos en Pórtico por Fecha")
        fig2.update_layout(width=900, height=500)
        fig2.update_xaxes(type='date', tickformat='%Y-%m-%d', dtick="D1")  # Configurar el formato del eje x y el intervalo de ticks
        fig2.update_yaxes(title_text='Total (m3)')  # Configurar el título del eje y
        fig2.update_traces(marker_color='#374752')  # Cambiar el color de las barras
        fig2.update_layout(
            xaxis_title_font=dict(size=20),  # Tamaño del título del eje x
            yaxis_title_font=dict(size=20),  # Tamaño del título del eje y
            xaxis=dict(tickfont=dict(size=18)),  # Tamaño de los valores del eje x
            yaxis=dict(tickfont=dict(size=18))   # Tamaño de los valores del eje y
        )
        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig2, use_container_width=True)
        patentes=df['Camion ID'].nunique()
        patentes


if funcion== "Análisis Excel Avance IX Etapa":
    st.title("📈 Análisis Avance IX Etapa")


    base_id='appOSiiFDWPc1n9hk'
    table_id='tblEKC7vFnnSxQ8JB'
    personal_access_token='patxqLGT8vqeIOi0S.0a930afb2c9db20d8ccfef398bbaafa7f3149a29f5c536e28aeae29f37fac516'
    
    def create_headers():
        headers = {
        'Authorization': 'Bearer ' + str(personal_access_token),
        'Content-Type': 'application/json',
        }
        return headers

    base_table_api_url = 'https://api.airtable.com/v0/{}/{}'.format(base_id, table_id)
    headers=create_headers()
    response = requests.get(base_table_api_url, headers=headers)
    if response.status_code == 200:
        # Convierte la respuesta en formato JSON a un diccionario
        data = response.json()
        
        # Extrae los registros de la respuesta
        records = data['records']
        
        # Crea una lista vacía para almacenar los datos de cada registro
        rows = []
        
        # Recorre cada registro y extrae los datos necesarios
        for record in records:
            fields = record['fields']
            rows.append(fields)

        # Crea el DataFrame utilizando la lista de registros
        df = pd.DataFrame(rows)
        


        # Define el orden de las columnas que deseas mantener
        column_order = ['SECCION', 'Total/Fecha', '11/25/23', '11/26/23', '11/27/23', '11/28/23', '11/29/23', '11/30/23', '12/1/23', '12/2/23', '12/3/23', '12/4/23', '12/5/23', '12/6/23', '12/7/23', '12/8/23', '12/9/23', '12/10/23', '12/11/23', '12/12/23', '12/13/23', '12/14/23', '12/15/23', '12/16/23', '12/17/23', '12/18/23', '12/19/23', '12/20/23', '12/21/23', '12/22/23', '12/23/23', '12/24/23', '12/25/23', '12/26/23', '12/27/23', '12/28/23', '12/29/23', '12/30/23', '12/31/23', '1/1/24', '1/2/24', '1/3/24', '1/4/24', '1/5/24', '1/6/24', '1/7/24', '1/8/24', '1/9/24', '1/10/24', '1/11/24', '1/12/24', '1/13/24', '1/14/24', '1/15/24', '1/16/24', '1/17/24', '1/18/24', '1/19/24', '1/20/24', '1/21/24', '1/22/24', '1/23/24', '1/24/24', '1/25/24', '1/26/24', '1/27/24', '1/28/24', '1/29/24', '1/30/24', '1/31/24', '2/1/24', '2/2/24', '2/3/24', '2/4/24', '2/5/24', '2/6/24', '2/7/24', '2/8/24', '2/9/24', '2/10/24', '2/11/24', '2/12/24', '2/13/24', '2/14/24', '2/15/24', '2/16/24', '2/17/24', '2/18/24', '2/19/24', '2/20/24', '2/21/24', '2/22/24', '2/23/24', '2/24/24', '2/25/24', '2/26/24', '2/27/24', '2/28/24','2/29/24' ,'3/1/24', '3/2/24', '3/3/24', '3/4/24', '3/5/24', '3/6/24', '3/7/24', '3/8/24', '3/9/24', '3/10/24', '3/11/24', '3/12/24', '3/13/24', '3/14/24', '3/15/24', '3/16/24', '3/17/24', '3/18/24', '3/19/24', '3/20/24', '3/21/24', '3/22/24', '3/23/24', '3/24/24', '3/25/24', '3/26/24', '3/27/24', '3/28/24', '3/29/24', '3/30/24', '3/31/24', '4/1/24', '4/2/24', '4/3/24', '4/4/24', '4/5/24', '4/6/24', '4/7/24', '4/8/24', '4/9/24', '4/10/24', '4/11/24', '4/12/24', '4/13/24', '4/14/24','4/15/24', '4/16/24', '4/17/24', '4/18/24', '4/19/24', '4/20/24', '4/21/24', '4/22/24', '4/23/24', '4/24/24', '4/25/24', '4/26/24', '4/27/24', '4/28/24', '4/29/24', '4/30/24', '5/1/24', '5/2/24', '5/3/24', '5/4/24', '5/5/24', '5/6/24', '5/7/24', '5/8/24', '5/9/24', '5/10/24', '5/11/24', '5/12/24', '5/13/24', '5/14/24', '5/15/24', '5/16/24', '5/17/24', '5/18/24', '5/19/24', '5/20/24', '5/21/24', '5/22/24', '5/23/24', '5/24/24','5/25/24',"5/26/24", "5/27/24", "5/28/24", "5/29/24", "5/30/24", "5/31/24", "6/1/24", "6/2/24", "6/3/24", "6/4/24", "6/5/24", "6/6/24", "6/7/24", "6/8/24", "6/9/24", "6/10/24", "6/11/24", "6/12/24", "6/13/24", "6/14/24", "6/15/24", "6/16/24", "6/17/24", "6/18/24", "6/19/24", "6/20/24", "6/21/24", "6/22/24", "6/23/24", "6/24/24", "6/25/24", "6/26/24", "6/27/24", "6/28/24", "6/29/24", "6/30/24", "7/1/24", "7/2/24", "7/3/24", "7/4/24", "7/5/24", "7/6/24", "7/7/24", "7/8/24", "7/9/24", "7/10/24", "7/11/24", "7/12/24", "7/13/24", "7/14/24", "7/15/24", "7/16/24", "7/17/24", "7/18/24", "7/19/24", "7/20/24", "7/21/24", "7/22/24", "7/23/24", "7/24/24", "7/25/24", "7/26/24", "7/27/24", "7/28/24", "7/29/24", "7/30/24", "7/31/24", "8/1/24", "8/2/24", "8/3/24", "8/4/24", "8/5/24", "8/6/24", "8/7/24", "8/8/24", "8/9/24", "8/10/24", "8/11/24", "8/12/24", "8/13/24", "8/14/24", "8/15/24", "8/16/24", "8/17/24", "8/18/24", "8/19/24", "8/20/24", "8/21/24", "8/22/24", "8/23/24", "8/24/24", "8/25/24", "8/26/24", "8/27/24", "8/28/24", "8/29/24", "8/30/24", "8/31/24", "9/1/24", "9/2/24", "9/3/24", "9/4/24", "9/5/24", "9/6/24", "9/7/24", "9/8/24", "9/9/24", "9/10/24", "9/11/24", "9/12/24", "9/13/24", "9/14/24", "9/15/24", "9/16/24", "9/17/24", "9/18/24", "9/19/24", "9/20/24", "9/21/24", "9/22/24", "9/23/24", "9/24/24", "9/25/24", "9/26/24", "9/27/24", "9/28/24", "9/29/24", "9/30/24", "10/1/24", "10/2/24", "10/3/24", "10/4/24", "10/5/24", "10/6/24", "10/7/24", "10/8/24", "10/9/24", "10/10/24", "10/11/24", "10/12/24", "10/13/24", "10/14/24", "10/15/24", "10/16/24", "10/17/24", "10/18/24", "10/19/24", "10/20/24", "10/21/24", "10/22/24", "10/23/24", "10/24/24", "10/25/24", "10/26/24", "10/27/24", "10/28/24", "10/29/24", "10/30/24", "10/31/24", "11/1/24", "11/2/24", "11/3/24", "11/4/24", "11/5/24", "11/6/24", "11/7/24", "11/8/24", "11/9/24", "11/10/24", "11/11/24", "11/12/24", "11/13/24", "11/14/24", "11/15/24", "11/16/24", "11/17/24", "11/18/24", "11/19/24", "11/20/24", "11/21/24", "11/22/24", "11/23/24", "11/24/24", "11/25/24", "11/26/24", "11/27/24", "11/28/24", "11/29/24", "11/30/24", "12/1/24", "12/2/24", "12/3/24", "12/4/24", "12/5/24", "12/6/24", "12/7/24", "12/8/24", "12/9/24", "12/10/24", "12/11/24", "12/12/24", "12/13/24", "12/14/24", "12/15/24", "12/16/24", "12/17/24", "12/18/24", "12/19/24", "12/20/24", "12/21/24", "12/22/24", "12/23/24", "12/24/24", "12/25/24", "12/26/24", "12/27/24", "12/28/24", "12/29/24", "12/30/24", "12/31/24", "1/1/25", "1/2/25", "1/3/25", "1/4/25", "1/5/25", "1/6/25", "1/7/25", "1/8/25", "1/9/25", "1/10/25", "1/11/25", "1/12/25", "1/13/25", "1/14/25", "1/15/25", "1/16/25", "1/17/25", "1/18/25", "1/19/25", "1/20/25", "1/21/25", "1/22/25", "1/23/25", "1/24/25", "1/25/25", "1/26/25", "1/27/25", "1/28/25", "1/29/25", "1/30/25", "1/31/25", "2/1/25", "2/2/25", "2/3/25", "2/4/25", "2/5/25", "2/6/25", "2/7/25", "2/8/25", "2/9/25", "2/10/25", "2/11/25", "2/12/25", "2/13/25", "2/14/25", "2/15/25", "2/16/25", "2/17/25", "2/18/25", "2/19/25", "2/20/25", "2/21/25", "2/22/25", "2/23/25", "2/24/25", "2/25/25", "2/26/25", "2/27/25", "2/28/25", "3/1/25", "3/2/25", "3/3/25", "3/4/25", "3/5/25", "3/6/25", "3/7/25", "3/8/25", "3/9/25", "3/10/25", "3/11/25", "3/12/25", "3/13/25", "3/14/25", "3/15/25", "3/16/25", "3/17/25", "3/18/25", "3/19/25", "3/20/25", "3/21/25", "3/22/25", "3/23/25", "3/24/25", "3/25/25", "3/26/25", "3/27/25", "3/28/25", "3/29/25", "3/30/25", "3/31/25", "4/1/25", "4/2/25", "4/3/25", "4/4/25", "4/5/25", "4/6/25", "4/7/25", "4/8/25", "4/9/25", "4/10/25", "4/11/25", "4/12/25", "4/13/25", "4/14/25", "4/15/25", "4/16/25", "4/17/25", "4/18/25", "4/19/25", "4/20/25", "4/21/25", "4/22/25", "4/23/25", "4/24/25", "4/25/25", "4/26/25", "4/27/25", "4/28/25", "4/29/25", "4/30/25", "5/1/25", "5/2/25", "5/3/25", "5/4/25", "5/5/25", "5/6/25", "5/7/25", "5/8/25", "5/9/25", "5/10/25", "5/11/25", "5/12/25", "5/13/25", "5/14/25", "5/15/25", "5/16/25", "5/17/25", "5/18/25", "5/19/25", "5/20/25", "5/21/25", "5/22/25", "5/23/25", "5/24/25", "5/25/25", "5/26/25", "5/27/25", "5/28/25", "5/29/25", "5/30/25", "5/31/25", "6/1/25", "6/2/25", "6/3/25", "6/4/25", "6/5/25", "6/6/25", "6/7/25", "6/8/25", "6/9/25", "6/10/25", "6/11/25", "6/12/25", "6/13/25", "6/14/25", "6/15/25", "6/16/25", "6/17/25", "6/18/25", "6/19/25", "6/20/25", "6/21/25", "6/22/25", "6/23/25", "6/24/25", "6/25/25", "6/26/25", "6/27/25", "6/28/25", "6/29/25", "6/30/25", "7/1/25", "7/2/25", "7/3/25", "7/4/25", "7/5/25", "7/6/25", "7/7/25", "7/8/25", "7/9/25", "7/10/25", "7/11/25", "7/12/25", "7/13/25", "7/14/25", "7/15/25", "7/16/25", "7/17/25", "7/18/25", "7/19/25", "7/20/25", "7/21/25", "7/22/25", "7/23/25", "7/24/25", "7/25/25", "7/26/25", "7/27/25", "7/28/25", "7/29/25", "7/30/25", "7/31/25", "8/1/25", "8/2/25", "8/3/25", "8/4/25", "8/5/25", "8/6/25", "8/7/25", "8/8/25", "8/9/25", "8/10/25", "8/11/25", "8/12/25", "8/13/25", "8/14/25", "8/15/25", "8/16/25", "8/17/25", "8/18/25", "8/19/25", "8/20/25", "8/21/25", "8/22/25", "8/23/25", "8/24/25", "8/25/25", "8/26/25", "8/27/25", "8/28/25", "8/29/25", "8/30/25", "8/31/25", "9/1/25", "9/2/25", "9/3/25", "9/4/25", "9/5/25", "9/6/25", "9/7/25", "9/8/25", "9/9/25", "9/10/25", "9/11/25", "9/12/25", "9/13/25", "9/14/25", "9/15/25", "9/16/25", "9/17/25", "9/18/25", "9/19/25", "9/20/25", "9/21/25", "9/22/25", "9/23/25", "9/24/25", "9/25/25", "9/26/25", "9/27/25", "9/28/25", "9/29/25", "9/30/25", "10/1/25", "10/2/25", "10/3/25", "10/4/25", "10/5/25", "10/6/25", "10/7/25", "10/8/25", "10/9/25", "10/10/25", "10/11/25", "10/12/25", "10/13/25", "10/14/25", "10/15/25", "10/16/25", "10/17/25", "10/18/25", "10/19/25", "10/20/25", "10/21/25", "10/22/25", "10/23/25", "10/24/25", "10/25/25", "10/26/25", "10/27/25", "10/28/25", "10/29/25", "10/30/25", "10/31/25", "11/1/25", "11/2/25", "11/3/25", "11/4/25", "11/5/25", "11/6/25", "11/7/25", "11/8/25", "11/9/25", "11/10/25", "11/11/25", "11/12/25", "11/13/25", "11/14/25", "11/15/25", "11/16/25", "11/17/25", "11/18/25", "11/19/25", "11/20/25", "11/21/25", "11/22/25", "11/23/25", "11/24/25", "11/25/25", "11/26/25", "11/27/25", "11/28/25", "11/29/25", "11/30/25", "12/1/25", "12/2/25", "12/3/25", "12/4/25", "12/5/25", "12/6/25", "12/7/25", "12/8/25", "12/9/25", "12/10/25", "12/11/25", "12/12/25", "12/13/25", "12/14/25", "12/15/25", "12/16/25", "12/17/25", "12/18/25", "12/19/25", "12/20/25", "12/21/25", "12/22/25", "12/23/25", "12/24/25", "12/25/25", "12/26/25", "12/27/25"]

        # Reindexa el DataFrame utilizando el orden de las columnas
        df = df.reindex(columns=column_order)
    fecha_actual = np.datetime64(datetime.now().date())
    mapeo = {
        'ZAMPEADO PRETIL PRINCIPAL T3 4.2.1.1.1': 'PRETIL PPAL EBMN',
        'RELLENO PRETIL PRINCIPAL T3 4.2.1.1.2': 'PRETIL PPAL EBMN',
        'RELLENO PRETIL PRINCIPAL T3 4.2.1.1.2 INTERNO': 'PRETIL PPAL EBMN',
        'ZAMPEADO PRETIL CONTORNO (ANILLO) EBMN 5.1.2.3.1': 'PRETIL PPAL EBMN',
        'RELLENO PRETIL CONTORNO (ANILLO) EBMN  5.1.2.3.2': 'PRETIL PPAL EBMN',
        'ZAMPEADO PRETIL PRINCIPAL T1 4.2.1.1.1': 'PRETIL PPAL EBMN',
        'RELLENO PRETIL PRINCIPAL T1 4.2.1.1.2': 'PRETIL PPAL EBMN',
        'RELLENO PLATAFORMAS 1,2,3,4,5 y 6 EBMN  5.1.2.2.2': 'PRETIL PPAL EBMN',
        'ZAMPEADO PLATAFORMAS EBMN 1,2,3,4,5 y 6  5.1.2.2.1': 'PRETIL PPAL EBMN',
        'RELLENO ZAMPEADO CAMINO PRETIL PRINCIPAL T1 4.2.1.1.1': 'CAMINOS ZAMPEADO',
        'BYPASS EBMN (ZAMPEADO)': 'CAMINOS ZAMPEADO',         
        'RELLENO AGUAS ABAJO MURO OESTE 1.4.1.3.1': 'MURO OESTE Y NO 1',
        'RELLENO AGUAS ABAJO MURO NOROESTE 1 1.3.1.2.1': 'MURO OESTE Y NO 1',
        'RELLENO MURO SUR TRAMO 1 1.5.1.3.1': 'MURO SUR',
        'ZAMPEADO PRETIL 2N-S 4.1.1.1.1': 'PRETIL 2 NS',
        'RELLENO PRETIL 2N-S':'PRETIL 2 NS',
        'Total': 'Total'
    }
    # Crear la nueva columna 'Unidad de Negocio'
    df['Secciones']=df['SECCION'].map(mapeo)
    # Definir la función para aplicar el factor de multiplicación
    def apply_factor(x):
        return (x * 14) / 20

    # Aplicar el factor de multiplicación a las columnas seleccionadas
    fechas_factor=['11/25/23', '11/26/23', '11/27/23', '11/28/23', '11/29/23', '11/30/23', '12/1/23', '12/2/23', '12/3/23', '12/4/23', '12/5/23', '12/6/23', '12/7/23', '12/8/23', '12/9/23', '12/10/23', '12/11/23', '12/12/23', '12/13/23', '12/14/23', '12/15/23', '12/16/23', '12/17/23', '12/18/23', '12/19/23', '12/20/23', '12/21/23', '12/22/23', '12/23/23', '12/24/23', '12/25/23', '12/26/23', '12/27/23', '12/28/23', '12/29/23', '12/30/23', '12/31/23', '1/1/24', '1/2/24', '1/3/24', '1/4/24', '1/5/24', '1/6/24', '1/7/24', '1/8/24', '1/9/24', '1/10/24', '1/11/24', '1/12/24', '1/13/24', '1/14/24', '1/15/24', '1/16/24', '1/17/24', '1/18/24', '1/19/24', '1/20/24', '1/21/24', '1/22/24', '1/23/24', '1/24/24', '1/25/24', '1/26/24', '1/27/24', '1/28/24', '1/29/24', '1/30/24', '1/31/24', '2/1/24', '2/2/24', '2/3/24', '2/4/24', '2/5/24', '2/6/24', '2/7/24', '2/8/24', '2/9/24', '2/10/24', '2/11/24', '2/12/24', '2/13/24']

    df[fechas_factor] = df[fechas_factor].apply(apply_factor)        


    df_limpieza = df.drop("SECCION", axis=1)
    df_limpieza = df_limpieza.groupby("Secciones").sum()
    df_limpieza = df_limpieza.reset_index()
    
    df_total = df_limpieza.loc[df_limpieza['Secciones'] == 'Total']
    columnas_suma = df_total.columns[2:]
    suma_total = df_total[columnas_suma].sum()
    suma_actual=int(suma_total.sum())
    
    # Crear un nuevo dataframe con la columna 'secciones'
    df_secciones = pd.DataFrame(df_limpieza['Secciones'])

    # Calcula la media móvil de 10 días para cada sección
    df_limpieza_media_movil = df_limpieza.iloc[:, 2:].rolling(window=10, axis=1).mean()


    # Concatenar el nuevo dataframe con 'df_limpieza_mediamovil'
    df_limpieza_media_movil = pd.concat([df_limpieza_media_movil, df_secciones], axis=1)
    #st.write(df_final)
    # Imprime el nuevo dataframe con la media móvil de 10 días

    # Obtener las secciones del DataFrame
    secciones = df_limpieza_media_movil["Secciones"].unique()

    # Crear un DataFrame para los valores y las fechas
    data = pd.DataFrame(columns=["Fecha", "Metros Cúbicos Compactados", "Seccion"])
    lista_fecha=[]

    for seccion in secciones:
        # Obtener los valores de la sección
        valores = df_limpieza_media_movil[df_limpieza_media_movil["Secciones"] == seccion].iloc[:, 2:-1]
        
        # Convertir las fechas a formato datetime
        fechas = valores.columns  # Las fechas ya están en el formato correcto
        
        # Crear un DataFrame con las fechas, valores y sección
        temp_df = pd.DataFrame({
            "Fecha": fechas,
            "Metros Cúbicos Compactados": valores.values[0],
            "Seccion": seccion
        })
        ultima_fecha = temp_df[temp_df["Metros Cúbicos Compactados"] != 0]["Fecha"].max()

        # Agregar el DataFrame temporal al DataFrame principal
        data = pd.concat([data, temp_df])
        lista_fecha.append(ultima_fecha)

    lista_fecha = max(lista_fecha)
    
    lista_fecha=pd.to_datetime(lista_fecha)
    
    

    # Convertir la columna de fechas a tipo datetime
    data["Fecha"] = pd.to_datetime(data["Fecha"])
    # Restar un día a las fechas
    fechas_ayer = fecha_actual - np.timedelta64(1, 'D')
    
    data = data[data["Fecha"] <= fechas_ayer]

    # Definir los colores deseados

    # Definir los colores para las rectas
    colores = ["#f4a700", "#374752", "#c8b499", "#bb5726", "#76151f"]
    data_total1 = data[data['Seccion'] == 'Total']
    # Graficar los valores utilizando plotly
    fig = px.line(data, x="Fecha", y="Metros Cúbicos Compactados", color="Seccion", title="Rellenos Compactados Media Móvil 10 días")

    # Modificar el color de las rectas
    for i, color in enumerate(colores):
        fig.data[i].line.color = color

    # Añadir una recta para el valor 3000
    fig.add_trace(go.Scatter(x=data["Fecha"], y=[3000] * len(data), mode="lines", name="Meta Diaria Frentes", line_color="#209eb0"))
    # Añadir una recta para el valor 15000
    fig.add_trace(go.Scatter(x=data["Fecha"], y=[15000] * len(data), mode="lines", name="Meta Diaria Total", line_color="#bb5726"))

    fig.update_layout(width=900, height=500)
    st.plotly_chart(fig, use_container_width=True)
    # Obtener las secciones del DataFrame
    secciones = df_limpieza_media_movil["Secciones"].unique()

    # Crear un DataFrame para los valores y las fechas
    data = pd.DataFrame(columns=["Fecha", "Metros Cúbicos Compactados", "Seccion"])

    # Obtener los valores y las fechas para cada sección
    for seccion in secciones:
        # Obtener los valores de la sección
        valores = df_limpieza_media_movil[df_limpieza_media_movil["Secciones"] == seccion].iloc[:, 2:-1]
        
        # Convertir las fechas a formato datetime
        fechas = valores.columns  # Las fechas ya están en el formato correcto
        
        # Crear un DataFrame con las fechas, valores y sección
        temp_df = pd.DataFrame({
            "Fecha": fechas,
            "Metros Cúbicos Compactados": valores.values[0],
            "Seccion": seccion
        })
        
        # Agregar el DataFrame temporal al DataFrame principal
        data = pd.concat([data, temp_df])

    # Convertir la columna de fechas a tipo datetime
    data["Fecha"] = pd.to_datetime(data["Fecha"])
    data = data[data["Fecha"] <= fechas_ayer]
    data_total = data[data['Seccion'] == 'Total']
    data = data[data['Seccion'] != 'Total']
    

    df_total = df_limpieza[df_limpieza["Secciones"] == "Total"]

    #PRUEBA
    valores = df_total[df_total["Secciones"] == seccion].iloc[:, 2:-1]
    
    # Convertir las fechas a formato datetime
    fechas = valores.columns  # Las fechas ya están en el formato correcto
    
    # Crear un DataFrame con las fechas, valores y sección
    temp_df = pd.DataFrame({
        "Fecha": fechas,
        "Metros Cúbicos Compactados": valores.values[0],
        "Seccion": seccion
    })
    import plotly.graph_objects as go

    # Agregar el DataFrame temporal al DataFrame principal
    data_total = temp_df
    data_total["Fecha"] = pd.to_datetime(data_total["Fecha"])
    data_total = data_total[data_total["Fecha"] <= fechas_ayer]
    #PRUEBA
    
    fig = px.bar(data, x="Fecha", y="Metros Cúbicos Compactados", color="Seccion", title="Rellenos Compactados Medias Móviles 10 días Apiladas", barmode="stack", color_discrete_sequence=["#f4a700", "#374752", "#c8b499", "#bb5726", "#76151f"])
    # Crear el gráfico de dispersión y asignarle un nombre para la leyenda
    # Crear el trace adicional utilizando go.Scatter() y asignarle un nombre para la leyenda
    scatter_trace = go.Scatter(
        x=data_total["Fecha"],
        y=data_total["Metros Cúbicos Compactados"],
        mode="markers",
        marker=dict(
            symbol="x",  # Cambia la forma de los puntos a cuadrados
            size=5,  # Cambia el tamaño de los puntos
            color="black"  # Cambia el color de los puntos
        ),
        name="Relleno Total Diario"
    )
    # Agregar el trace adicional al gráfico de barras
    fig.add_trace(scatter_trace)

    fig.update_layout(width=900, height=500)

    #data_total
    valor_mas_reciente1 = data_total1.loc[data_total1['Fecha'].idxmax(), 'Metros Cúbicos Compactados']
    valor_mas_reciente = data_total.loc[data_total['Fecha'].idxmax(), 'Metros Cúbicos Compactados']





    #alor_mas_reciente
    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**m³ Geométricos:**")

    col2,col4,col5,col3=st.columns(4)
    col2.metric(label="Acumulados", value=format(suma_actual,',').replace(',', '.'))
    maximo = data_total['Metros Cúbicos Compactados'].max()
    col3.metric(label="Máximo Alcanzado en un día", value=format(int(maximo),',').replace(',', '.'))
    col4.metric(label='Último Relleno', value=format(int(valor_mas_reciente),',').replace(',', '.'))
    col5.metric(label='Última Media Móvil', value=format(int(valor_mas_reciente1),',').replace(',', '.'))



    style_metric_cards()
    #col2.metric(label="Acumulado Histórico",value=suma_actual,delta=str(round(suma_actual/(suma_diciembre)*100,1))+"% de Cumplimiento")

    #col3.metric(label="Total por Depositar",value=suma_diciembre-suma_actual,delta=str(-round(((suma_diciembre-suma_actual)/suma_diciembre)*100,1))+"% Restante")

    #Tablas que pidió Camilo Sacar 
    #st.markdown("**Producción Agrupada por Sección**")
    #st.write(df_limpieza)
    #st.markdown("**Media Movil**")
    #st.write(df_limpieza_media_movil)
if funcion=="Tiempo en Geocercas":
    fig = go.Figure()
    figw = go.Figure()

    color_boxplot = '#6e4d48'
    color_promedio = 'black'
    font_size_promedio = 16
    st.sidebar.title('Cargar archivo')
    uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV o XLSX", type=['csv', 'xlsx'])
    df = pd.read_excel(uploaded_file)
    #df=df[df['duración_permanencia_geocerca__min_']<40]
    #df=df[df['duración_permanencia_geocerca__min_']>1]
    color_boxplot='#f4a700'
    for lugar_descarga in sorted(df['geocerca'].unique(), reverse=True):

        data = df[df['geocerca'] == lugar_descarga]['duración_permanencia_geocerca__min_']
        fig.add_trace(go.Box(y=data, name=lugar_descarga, marker_color=color_boxplot, showlegend=False))

        # Agregar anotaciones
        media = round(data.median(), 1)
        fig.add_annotation(x=lugar_descarga, y=media+5, text=str(media), showarrow=False, font=dict(color=color_promedio, size=font_size_promedio))


    df = df[~(df['geocerca'] == 'Parqueo Geocerca') | (df['duración_permanencia_geocerca__min_'] < 40)]
    df=df[df['duración_permanencia_geocerca__min_']<90]
    #df=df[df['duración_permanencia_geocerca__min_']>3]


    # Filtrar los datos
    df_plataforma_5 = df[(df['duración_permanencia_geocerca__min_'] > 5) & (df['geocerca'] == 'Plataforma 5')]
    df_no_plataforma_5 = df[(df['duración_permanencia_geocerca__min_'] > 3) & (df['geocerca'] != 'Plataforma 5')]
    df_final = pd.concat([df_plataforma_5, df_no_plataforma_5])

    color_boxplot = '#374752'
    color_promedio = '#201F1F'  # Color para la línea del promedio
    font_size_promedio = 18

    figw = go.Figure()

    # Obtener la lista de geocercas ordenadas
    geocercas = sorted(df_final['geocerca'].unique(), reverse=True)

    for i, lugar_descarga in enumerate(geocercas):
        data = df_final[df_final['geocerca'] == lugar_descarga]['duración_permanencia_geocerca__min_']
        
        # Agregar el boxplot
        figw.add_trace(go.Box(y=data, name=lugar_descarga, marker_color=color_boxplot, showlegend=False))
        
        # Calcular el promedio
        promedio = round(data.mean(), 1)
        
        # Agregar línea horizontal para el promedio
        figw.add_shape(
            type="line",
            x0=i - 0.4,  # Ajustar la posición x0
            y0=promedio,
            x1=i + 0.4,  # Ajustar la posición x1
            y1=promedio,
            line=dict(color=color_promedio, width=3, dash="dash"),  # Aumentar el grosor de la línea
            xref="x",
            yref="y"
        )

        # Agregar anotaciones para el promedio
        figw.add_annotation(x=i, y=promedio + 1, text=str(promedio), showarrow=False, font=dict(color=color_promedio, size=font_size_promedio))
        
        # Agregar anotación para la cantidad de datos
        cantidad_datos = len(data)
        figw.add_annotation(x=i, y=min(data) - 2, text=f'{cantidad_datos} Registros', showarrow=False, font=dict(size=18))

    # Configuración del layout
    figw.update_layout(title='Box Plot Tiempo en Geocerca por Frentes',
                    xaxis_title='Frente',
                    yaxis_title='Permanencia en Geocerca (min)',
                    width=600,
                    height=600)

    # Agregar línea invisible para la leyenda
    figw.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color=color_promedio, width=3, dash='dash'),
        showlegend=True,
        name='Promedio'
    ))

    figw.update_layout(
        xaxis_title_font=dict(size=20),
        yaxis_title_font=dict(size=20),
        xaxis=dict(tickfont=dict(size=18)),
        yaxis=dict(tickfont=dict(size=18)),
        legend=dict(font=dict(size=16))

    )

    # Mostrar el gráfico
    st.plotly_chart(figw, use_container_width=True)




    # Supongamos que df_final es tu DataFrame que contiene las columnas 'fecha_inicio', 'patente' y 'geocerca'

    # Asegúrate de que 'fecha_inicio' sea de tipo datetime
    df_final['fecha_inicio'] = pd.to_datetime(df_final['fecha_inicio']).dt.date

    # Agrupar por fecha y geocerca, contando las patentes únicas
    df_agrupado = df_final.groupby(['fecha_inicio', 'geocerca'])['patente'].nunique().reset_index()
    color_mapping = {
        "Muro Norte": "#209eb0",
        "Muro Sur": "#76151f",
        "Muro Oeste": "#bb5726",
        "Pretil Principal": "#209eb0",
        "Pretil 6.1-2": "#c8b499",
        "Pretil 2A-2B": "#374752",
        "Pretil Principal T1": "#c8b499",
        "Plataforma 1":"#f4a700",
        "Plataforma 5":"#004c4e"
    }

    # Crear el gráfico de barras
    fig = px.bar(df_agrupado, 
                x='fecha_inicio', 
                y='patente', 
                color='geocerca', 
                barmode='group',
                title='Patentes Únicas por Geocerca y Fecha',
                labels={'patente': 'Cantidad de Patentes Únicas', 'fecha_inicio': 'Fecha'},
                color_discrete_map=color_mapping)

    # Personalizar el diseño
    fig.update_layout(xaxis_title='Fecha',
                    yaxis_title='Cantidad de Patentes Únicas',
                    xaxis_tickformat='%Y-%m-%d',  # Formato de fecha
                    width=800,
                    height=600)
    fig.update_layout(
        xaxis_title_font=dict(size=20),  # Tamaño del título del eje x
        yaxis_title_font=dict(size=20),  # Tamaño del título del eje y
        xaxis=dict(tickfont=dict(size=18)),  # Tamaño de los valores del eje x
        yaxis=dict(tickfont=dict(size=18)),
        legend=dict(font=dict(size=18))
   # Tamaño de los valores del eje y
    )

    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)
        # Definir un mapeo de colores para cada lugar_descarga

