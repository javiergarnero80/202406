import pandas as pd
import folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from pathlib import Path
from tkinter import filedialog, messagebox, Tk, Button, Label, Toplevel, BooleanVar
from tkinter.ttk import Progressbar
import webbrowser
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Constantes y Configuraciones
GEOLOCATOR_USER_AGENT = "my_geocoder_app"
GEOLOCATOR_TIMEOUT = 10  # En segundos
OUTPUT_DIR = Path.home() / "Desktop" / "GeocoderApp"  # Ruta donde se guardarán los archivos
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Crea el directorio si no existe

# Inicialización del Geolocalizador con un limitador de tasa de peticiones
geolocator = Nominatim(user_agent=GEOLOCATOR_USER_AGENT, timeout=GEOLOCATOR_TIMEOUT)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

class GeocoderApp:
    """Clase principal para la aplicación de geocodificación."""

    def __init__(self, master):
        """Inicializa la interfaz gráfica y los atributos de la clase."""
        self.master = master
        self.master.title("Geocodificador de Direcciones")

        self.label = Label(master, text="Cargar CSV y geocodificar direcciones")
        self.label.grid(columnspan=2, row=0)
        
        self.load_button = Button(master, text="Cargar CSV", command=self.cargar_archivo)
        self.load_button.grid(row=1, column=0)
        
        self.geocode_button = Button(master, text="Geocodificar", command=self.geocodificar_dataframe)
        self.geocode_button.grid(row=1, column=1)
        
        self.map_button = Button(master, text="Generar Mapa", command=self.generar_mapa)
        self.map_button.grid(row=2, column=0, columnspan=2)
        
        self.sum_button = Button(master, text="Sumar Columnas", command=self.sumar_columnas)
        self.sum_button.grid(row=3, column=0, columnspan=2)
        
        self.analisis_temporal_button = Button(master, text="Análisis Temporal", command=self.analisis_temporal)
        self.analisis_temporal_button.grid(row=4, column=0, columnspan=2)
        
        self.analisis_geoespacial_button = Button(master, text="Análisis Geoespacial", command=self.analisis_geoespacial)
        self.analisis_geoespacial_button.grid(row=5, column=0, columnspan=2)
        
        self.analisis_correlacion_button = Button(master, text="Análisis de Correlación", command=self.analisis_correlacion)
        self.analisis_correlacion_button.grid(row=6, column=0, columnspan=2)
        
        self.modelos_predictivos_button = Button(master, text="Modelos Predictivos", command=self.modelos_predictivos)
        self.modelos_predictivos_button.grid(row=7, column=0, columnspan=2)
        
        self.analisis_riesgos_button = Button(master, text="Análisis de Riesgos", command=self.analisis_riesgos)
        self.analisis_riesgos_button.grid(row=8, column=0, columnspan=2)
        
        self.df = pd.DataFrame()
        self.mapa_generado = False
        self.errores_geocodificacion = []
        self.cancelar_geocodificacion = BooleanVar(value=False)

    def cargar_archivo(self):
        """Carga un archivo CSV y lo lee en un DataFrame de pandas."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            encodings = ['utf-8', 'latin1']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(file_path, encoding=encoding)
                    self.df.columns = self.df.columns.astype(str).str.strip().str.lower()  # Normalizar nombres de columnas
                    # Corregir nombre de columna con problemas de codificación
                    if 'ï»¿pais' in self.df.columns:
                        self.df.rename(columns={'ï»¿pais': 'pais'}, inplace=True)
                    messagebox.showinfo("Archivo Cargado", f"Archivo CSV cargado con éxito usando codificación {encoding}.")
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            else:
                messagebox.showerror("Error al cargar archivo", "No se pudo cargar el archivo CSV con las codificaciones probadas.")
            
    def geocodificar_dataframe(self):
        """Geocodifica las direcciones contenidas en el DataFrame."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return
        
        # Normalizar nombres de columnas esperados
        columns_mapping = {
            'departamento': ['departamento', 'depto', 'dept'],
            'provincia': ['provincia', 'prov', 'provincia/estado'],
            'pais': ['pais', 'país', 'country']
        }
        
        for key, possible_names in columns_mapping.items():
            for name in possible_names:
                if name in self.df.columns:
                    self.df.rename(columns={name: key}, inplace=True)
                    break
        
        required_columns = ['departamento', 'provincia', 'pais']
        for col in required_columns:
            if col not in self.df.columns:
                messagebox.showerror("Error", f"El archivo CSV no contiene la columna requerida: {col}")
                return
        
        self.df['Direccion'] = self.df.apply(self.concatenar_direccion, axis=1)

        # Crear ventana de progreso
        progress_window = Toplevel(self.master)
        progress_window.title("Progreso de Geocodificación")
        Label(progress_window, text="Geocodificando direcciones...").pack(pady=10)
        progress_bar = Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
        progress_bar.pack(pady=20)
        progress_bar["maximum"] = len(self.df)

        progress_label = Label(progress_window, text="0% completado")
        progress_label.pack(pady=5)

        # Botón de cancelar
        cancel_button = Button(progress_window, text="Cancelar", command=lambda: self.cancelar_geocodificacion.set(True))
        cancel_button.pack(pady=5)

        self.errores_geocodificacion.clear()
        self.cancelar_geocodificacion.set(False)

        # Paralelizar la geocodificación
        self.df['Latitude'] = None
        self.df['Longitude'] = None
        self.df['GeocodedAddress'] = None
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_address = {executor.submit(self.obtener_location, direccion): i for i, direccion in enumerate(self.df['Direccion'])}
            for future in as_completed(future_to_address):
                if self.cancelar_geocodificacion.get():
                    messagebox.showinfo("Geocodificación Cancelada", "La geocodificación ha sido cancelada.")
                    break

                i = future_to_address[future]
                try:
                    location = future.result()
                    if location:
                        self.df.at[i, 'Latitude'] = location.latitude
                        self.df.at[i, 'Longitude'] = location.longitude
                        self.df.at[i, 'GeocodedAddress'] = location.address
                except Exception as e:
                    self.errores_geocodificacion.append((self.df.at[i, 'Direccion'], str(e)))
                    print(f"Error geocodificando la dirección '{self.df.at[i, 'Direccion']}': {str(e)}")
                progress_bar["value"] += 1
                progress_percentage = (progress_bar["value"] / len(self.df)) * 100
                progress_label.config(text=f"{progress_percentage:.2f}% completado")
                progress_window.update_idletasks()

        progress_window.destroy()
        
        if self.errores_geocodificacion:
            error_messages = "\n".join([f"{address}: {error}" for address, error in self.errores_geocodificacion])
            messagebox.showerror("Errores de Geocodificación", f"Se produjeron errores al geocodificar las siguientes direcciones:\n{error_messages}")
        elif not self.cancelar_geocodificacion.get():
            messagebox.showinfo("Geocodificación", "Geocodificación completada con éxito.")
        
        self.df.drop(columns=['Direccion'], inplace=True)
        self.mapa_generado = False

    @staticmethod
    def concatenar_direccion(row):
        """Concatena las columnas de dirección para formar una dirección completa."""
        return f"{row['departamento']}, {row['provincia']}, {row['pais']}"

    @staticmethod
    def obtener_location(direccion):
        """Obtiene la ubicación (latitud y longitud) para una dirección dada."""
        try:
            location = geocode(direccion)
            if location:
                return location
            else:
                return None
        except GeocoderTimedOut:
            raise Exception("El tiempo de espera para la geocodificación se agotó.")
        except GeocoderServiceError as e:
            raise Exception(f"Error en el servicio de geocodificación: {str(e)}")
        except Exception as e:
            raise Exception(f"Error desconocido durante la geocodificación: {str(e)}")

    def generar_mapa(self):
        """Genera un mapa con los puntos geocodificados y lo guarda en un archivo HTML."""
        if self.df.empty or 'Latitude' not in self.df or 'Longitude' not in self.df:
            messagebox.showwarning("Advertencia", "Realice la geocodificación antes de generar el mapa.")
            return

        if not self.mapa_generado:  
            lat_mean = self.df['Latitude'].dropna().mean()
            lon_mean = self.df['Longitude'].dropna().mean()
            mapa = folium.Map(location=[lat_mean, lon_mean], zoom_start=5)
            
            for lat, lon, address in zip(self.df['Latitude'], self.df['Longitude'], self.df['GeocodedAddress']):
                if lat and lon and address:
                    folium.Marker(
                        [lat, lon],
                        popup=f"{address}\n({lat}, {lon})"
                    ).add_to(mapa)
            
            map_file_path = OUTPUT_DIR / "mapa.html"
            mapa.save(str(map_file_path))
            self.mapa_generado = True
            
            messagebox.showinfo("Mapa Generado", f"Se ha generado el mapa en la ubicación {str(map_file_path)}.")
            webbrowser.open(str(map_file_path), new=2)

    def sumar_columnas(self):
        """Suma las columnas específicas del DataFrame y muestra el resultado."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return
        
        try:
            total = self.df[['sup_sembrada', 'sup_cosechada', 'produccion', 'rendimiento']].sum(numeric_only=True)
            messagebox.showinfo("Suma de Columnas", f"Suma de columnas:\n{total}")
        except Exception as e:
            messagebox.showerror("Error al sumar columnas", f"No se pudo sumar las columnas: {str(e)}")

    def analisis_temporal(self):
        """Realiza un análisis temporal de la producción."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        if 'campaña' not in self.df.columns:
            messagebox.showwarning("Advertencia", "La columna 'campaña' no existe en el DataFrame.")
            return

        try:
            self.df['campaña'] = self.df['campaña'].apply(lambda x: int(x.split('/')[0]))  # Convertir a año inicial
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo convertir la columna 'campaña': {str(e)}")
            return

        self.df.set_index('campaña', inplace=True)
        self.df.groupby('campaña')['produccion'].sum().plot()
        plt.title('Tendencia de Producción a lo largo del Tiempo')
        plt.xlabel('Campaña')
        plt.ylabel('Producción')
        plt.show()

    def analisis_geoespacial(self):
        """Realiza un análisis geoespacial de la producción."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        if 'Longitude' not in self.df.columns or 'Latitude' not in self.df.columns:
            messagebox.showerror("Error", "El DataFrame no contiene las columnas 'Longitude' y 'Latitude'.")
            return
        
        gdf = gpd.GeoDataFrame(self.df, geometry=gpd.points_from_xy(self.df['Longitude'], self.df['Latitude']))
        gdf.plot(column='produccion', cmap='OrRd', legend=True)
        plt.title('Mapa de Producción Agrícola')
        plt.show()

    def analisis_correlacion(self):
        """Realiza un análisis de correlación entre las columnas del DataFrame."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return
        
        # Filtrar solo columnas numéricas
        numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            messagebox.showerror("Error", "No hay columnas numéricas en el DataFrame para realizar el análisis de correlación.")
            return

        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title('Matriz de Correlación')
        plt.show()

    def modelos_predictivos(self):
        """Crea modelos predictivos de la producción agrícola."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        required_columns = ['sup_sembrada', 'sup_cosechada', 'rendimiento']
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            messagebox.showerror("Error", f"Faltan las siguientes columnas requeridas en el DataFrame: {', '.join(missing_columns)}")
            return

        X = self.df[required_columns]
        y = self.df['produccion']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        plt.scatter(y_test, y_pred)
        plt.xlabel("Valores Reales")
        plt.ylabel("Predicciones")
        plt.title("Predicciones vs Valores Reales")
        plt.show()

        # Mostrar métricas de desempeño
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        messagebox.showinfo("Desempeño del Modelo", f"MSE: {mse:.2f}\nR2: {r2:.2f}")

    def analisis_riesgos(self):
        """Realiza un análisis de riesgos utilizando simulaciones."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        required_columns = ['sup_sembrada', 'sup_cosechada', 'rendimiento']
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            messagebox.showerror("Error", f"Faltan las siguientes columnas requeridas en el DataFrame: {', '.join(missing_columns)}")
            return

        simulations = 1000
        predictions = []

        X = self.df[required_columns]
        y = self.df['produccion']
        model = LinearRegression()
        model.fit(X, y)

        for _ in range(simulations):
            simulated_sembrada = np.random.normal(self.df['sup_sembrada'].mean(), self.df['sup_sembrada'].std(), len(self.df))
            simulated_cosechada = np.random.normal(self.df['sup_cosechada'].mean(), self.df['sup_cosechada'].std(), len(self.df))
            simulated_rendimiento = np.random.normal(self.df['rendimiento'].mean(), self.df['rendimiento'].std(), len(self.df))
            simulated_production = model.predict(np.column_stack([simulated_sembrada, simulated_cosechada, simulated_rendimiento]))
            predictions.append(simulated_production)

        predictions = np.array(predictions)
        plt.hist(predictions.mean(axis=1), bins=50)
        plt.title('Distribución de Producción Simulada')
        plt.xlabel('Producción')
        plt.ylabel('Frecuencia')
        plt.show()

def main():
    """Función principal para iniciar la aplicación."""
    root = Tk()
    app = GeocoderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()