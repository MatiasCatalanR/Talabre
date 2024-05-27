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
st.sidebar.markdown("<hr style='border:2.5px solid white'> </hr>", unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: left; color: white;'>Unidad de Control Operativo</h1>", unsafe_allow_html=True)
funcion=st.sidebar.selectbox("Seleccione una Función",["Reporte Inicio-Término Turno","Transgresiones Historicas","Programación Rellenos","MDG 2024"])

url_despacho='https://icons8.com/icon/21183/in-transit'





# Ahora 'data' contiene la respuesta de la API en formato JSON

if funcion=="Reporte Inicio-Término Turno":
    st.title('📊 Análisis de Relleno al Inicio y Término del Turno Diurno')
    # Obtener la fecha seleccionada por el usuario
    #selected_date = st.sidebar.date_input("Seleccione una fecha")

    dias_a_restar = 10

    import datetime
    import streamlit as st

    import datetime
    import streamlit as st

    # Obtén la fecha de hoy y la fecha de hace 7 días
    hoy = datetime.date.today()
    hace_siete_dias = hoy - datetime.timedelta(days=9)

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
    # Credenciales para la autenticación básica
    username = "talabre"
    password = "cosmos_talabre_2024"

    # Realizar la petición GET a la API
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    # Verificar si la petición fue exitosa

    response2 = requests.get(url2, auth=HTTPBasicAuth(username, password))
    
    if response.status_code == 200 and response2.status_code == 200:
        # Convertir la respuesta a JSON
        data = response.json()
        df_ciclo=pd.DataFrame(data)
        data2 = response2.json()
        df_horometro=pd.DataFrame(data2)
        #st.write(df_horometro)
        if st.sidebar.checkbox('Mostrar base de datos'):
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
        col5.metric(label="m³ Transportados",value=total_ciclos*20)
        col6.metric(label="Total Ciclos",value=total_ciclos)
        col7.metric(label="Camiones Operativos",value=camiones_totales)
        st.metric(label="m³ Geométricos",value=int((total_ciclos*20)/1.42))
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



        # Convertir 'fin_descarga' a datetime y redondear a la hora más cercana
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
        st_echarts(options=options, height="400px")










        m3_km_transportado=int(m3_transportados)/int(km_recorridos)
        df['kph_mean']=df['kph_mean'].str.replace(',', '.').astype(float)
        df['kph_mean'] = pd.to_numeric(df['kph_mean'])
        df_velocidad = df[df['kph_mean'] > 15]

        #st.write(df_velocidad)
        velocidad_promedio=round(df_velocidad['kph_mean'].mean(),2)
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
        for (fecha, patente,tiempo_carga), group in df_carguio.groupby(['fecha', 'patente','tiempo_carga__min_']):
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



        # Convertir 'inicio_turno' a segundos
        #st.write(turno_diurno_df)
        
        turno_diurno_df['inicio_turno_segundos'] = turno_diurno_df['inicio_turno'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

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
        col1, col2,col3,col4=st.columns(4)
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
            entrada_carga_df=inicio_carga_df[['fecha','patente','inicio_carga']].copy()
            
            st.write(entrada_carga_df)

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
            fin_cargas_df=fin_carga_df[['fecha','patente','fin_carga']].copy()

            st.write(fin_cargas_df)
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

        # Muestra el gráfico en Streamlit
        st.plotly_chart(fig)

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



        histograma = df[['t_cola_carga', 'tiempo_carga__min_', 'transito_cargado__min_', 't_cola_descarga', 'tiempo_descarga__min_', 'transito_descargado__min_', 'tiempo_ciclo__min_', 'kph_mean', 'kph_max', 'distancia_recorrida__km_', 'tiempo_demoras_min']]

        # Set Seaborn style 
        sns.set_style("darkgrid") 
        # Asumiendo que df es tu DataFrame original
        df_limpieza = df.dropna()
        #histograma=histograma.apply(lambda col:pd.to_numeric(col, errors='coerce'))
        # Aplicar la conversión a cadenas de texto a todos los elementos del dataframe
        histograma = histograma.apply(lambda x: x.astype(str))

        # Aplicar el reemplazo de comas por puntos a todos los elementos del dataframe
        histograma = histograma.applymap(lambda x: x.replace(',', '.'))


        histograma = histograma.apply(lambda x: x.str.replace('%','').astype(np.float64))

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

   

    
            
    if funcion=="Transgresiones Historicas":
        st.sidebar.title('Cargar archivo')
        uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV o XLSX", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            st.write(df)
            data = df[['apellido', 'nombre']].value_counts().reset_index()
            data.columns = ['apellido', 'nombre', 'count']
            data_grafico = data[data['apellido'] != 'Sin registro']
            data_grafico = data_grafico[data_grafico['apellido'] != 'Sin Registro']


            
            pie = (
                Pie()
                .add("", [list(z) for z in zip(data_grafico['apellido'], data_grafico['count'])], radius=["40%", "75%"])
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
                                text=f"{len(data_grafico)} Infractores",
                                font="bold 17px Microsoft YaHei",
                                graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(fill="#333")
                            )
                        )
                    ]
                )
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}"), # Muestra solo el apellido
                                tooltip_opts=opts.TooltipOpts(formatter="{b}: {c} ({d}%)")) # Muestra el número y el porcentaje solo al pasar el mouse por encima
            )
            st.markdown("**Transgresiones de Velocidad por Operador**")
            # Mostrar el gráfico en Streamlit
            st_pyecharts(pie)
            st.write(data)


if funcion=="Programación Rellenos":

    st.title("🧮 Programación Rellenos")

    st.sidebar.title('Cargar archivo')
    uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV o XLSX", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        #st.write(df)

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

        # Convertir la columna 'Fecha' a formato de texto y extraer el mes y el año
        new_df['Fecha'] = pd.to_datetime(new_df['Fecha']).dt.strftime('%Y-%m')

        # Calcular la Metros Cúbicos para cada 'ÁREA' y 'Fecha'
        new_df = new_df.groupby(['ÁREA', 'Fecha'])['Metros Cúbicos'].sum().reset_index()

        # Calcular el total para cada 'Fecha'
        total_df = new_df.groupby('Fecha')['Metros Cúbicos'].sum().reset_index()
        total_df.columns = ['Fecha', 'Total']

        # Unir el dataframe original con los totales
        new_df = pd.merge(new_df, total_df, on='Fecha')
        
        # Ordena el DataFrame por 'Metros Cúbicos VP' en orden descendente
        new_df = new_df.sort_values('ÁREA')

        # Crea un diccionario que mapea cada 'ÁREA' a un color
        color_dict = {area: colormap[i % len(colormap)] for i, area in enumerate(new_df['ÁREA'].unique())}
        st.markdown("**Metros Cúbicos Mensuales a Transportar por Área**")
        # Crea el gráfico de barras apiladas
        chart = alt.Chart(new_df).mark_bar().encode(
            x='Fecha:N',
            y='Metros Cúbicos:Q',
            color=alt.Color('ÁREA:N', scale=alt.Scale(scheme="cividis")),
            tooltip=[
                'Fecha:N', 
                alt.Tooltip('ÁREA:N'),
                alt.Tooltip('Metros Cúbicos:Q', format=',d'),
                alt.Tooltip('Total:Q', format=',d')
            ],
            order=alt.Order(
            'Fecha:N',
            sort='ascending'
            )
        ).properties(
            width=800,
            height=500
        )
        chart

    else:
        st.markdown("Por favor Carga la Programación de Rellenos")

if funcion== "MDG 2024":

    st.title("📚 MDG 2024")

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
        
