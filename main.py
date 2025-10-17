from flask import Flask, render_template, request
from pulp import *

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def optimizar_bauxita():
    # Inicializar variables
    costo_total = None
    plantas_abiertas = []
    
    if request.method == "POST":
        # Obtener los 4 costos del formulario
        costo_B = int(request.form.get("costo_B"))
        costo_C = int(request.form.get("costo_C"))
        costo_D = int(request.form.get("costo_D"))
        costo_E = int(request.form.get("costo_E"))
        
        print("Costos recibidos:")
        print("Planta B:", costo_B)
        print("Planta C:", costo_C)
        print("Planta D:", costo_D)
        print("Planta E:", costo_E)
        
        # EJECUTAR EL MODELO DE OPTIMIZACIÓN
        modelo = LpProblem("La_bauxita", LpMinimize)
        
        # Variables de decisión
        X_ij = LpVariable.dicts("X", (['A','B','C'], ['B','C','D','E']), lowBound=0)
        Y_jk = LpVariable.dicts("Y", (['B','C','D','E'], ['D','E']), lowBound=0)
        W_j = LpVariable.dicts("W", ['B','C','D','E'], cat='Binary')
        
        # Función objetivo (USANDO LOS COSTOS INGRESADOS)
        Costo_explotacion = (
            420*(X_ij['A']['B'] + X_ij['A']['C'] + X_ij['A']['D'] + X_ij['A']['E']) +
            360*(X_ij['B']['B'] + X_ij['B']['C'] + X_ij['B']['D'] + X_ij['B']['E']) +
            540*(X_ij['C']['B'] + X_ij['C']['C'] + X_ij['C']['D'] + X_ij['C']['E'])
        )
        
        Costo_produccion = (
            330*(Y_jk['B']['D'] + Y_jk['B']['E']) +
            320*(Y_jk['C']['D'] + Y_jk['C']['E']) +
            380*(Y_jk['D']['D'] + Y_jk['D']['E']) +
            240*(Y_jk['E']['D'] + Y_jk['E']['E'])
        )
        
        Costo_procesamiento = (
            8500*(Y_jk['B']['D'] + Y_jk['C']['D'] + Y_jk['D']['D'] + Y_jk['E']['D']) +
            5200*(Y_jk['B']['E'] + Y_jk['C']['E'] + Y_jk['D']['E'] + Y_jk['E']['E'])
        )
        
        Costo_transporte_bauxita = (
            400*X_ij['A']['B'] + 2100*X_ij['A']['C'] + 510*X_ij['A']['D'] + 1920*X_ij['A']['E'] +
            10*X_ij['B']['B'] + 630*X_ij['B']['C'] + 220*X_ij['B']['D'] + 1510*X_ij['B']['E'] +
            1630*X_ij['C']['B'] + 10*X_ij['C']['C'] + 620*X_ij['C']['D'] + 940*X_ij['C']['E']
        )

        Costo_transporte_alumina = (
            220*Y_jk['B']['D'] + 620*Y_jk['C']['D'] + 1465*Y_jk['E']['D'] +
            1510*Y_jk['B']['E'] + 940*Y_jk['C']['E'] + 1615*Y_jk['D']['E'] 
        )   
        
        # ✅ USAMOS LOS COSTOS INGRESADOS POR EL USUARIO
        Costo_fijo_plantas = (
            costo_B * W_j['B'] + 
            costo_C * W_j['C'] + 
            costo_D * W_j['D'] + 
            costo_E * W_j['E']
        )

        modelo += (
            Costo_explotacion + 
            Costo_produccion + 
            Costo_procesamiento + 
            Costo_transporte_bauxita + 
            Costo_transporte_alumina + 
            Costo_fijo_plantas
        )

        # Restricciones (las mismas)
        modelo += X_ij['A']['B'] + X_ij['A']['C'] + X_ij['A']['D'] + X_ij['A']['E'] <= 36000
        modelo += X_ij['B']['B'] + X_ij['B']['C'] + X_ij['B']['D'] + X_ij['B']['E'] <= 52000
        modelo += X_ij['C']['B'] + X_ij['C']['C'] + X_ij['C']['D'] + X_ij['C']['E'] <= 28000

        modelo += X_ij['A']['B'] + X_ij['B']['B'] + X_ij['C']['B'] <= 40000 * W_j['B']
        modelo += X_ij['A']['C'] + X_ij['B']['C'] + X_ij['C']['C'] <= 20000 * W_j['C']
        modelo += X_ij['A']['D'] + X_ij['B']['D'] + X_ij['C']['D'] <= 30000 * W_j['D']
        modelo += X_ij['A']['E'] + X_ij['B']['E'] + X_ij['C']['E'] <= 80000 * W_j['E']

        modelo += Y_jk['B']['D'] + Y_jk['C']['D'] + Y_jk['D']['D'] + Y_jk['E']['D'] <= 4000
        modelo += Y_jk['B']['E'] + Y_jk['C']['E'] + Y_jk['D']['E'] + Y_jk['E']['E'] <= 7000
        
        modelo += 0.4 * (Y_jk['B']['D'] + Y_jk['C']['D'] + Y_jk['D']['D'] + Y_jk['E']['D']) == 1000
        modelo += 0.4 * (Y_jk['B']['E'] + Y_jk['C']['E'] + Y_jk['D']['E'] + Y_jk['E']['E']) == 1200

        modelo += 0.060 * X_ij['A']['B'] + 0.080 * X_ij['B']['B'] + 0.062 * X_ij['C']['B'] == Y_jk['B']['D'] + Y_jk['B']['E']
        modelo += 0.060 * X_ij['A']['C'] + 0.080 * X_ij['B']['C'] + 0.062 * X_ij['C']['C'] == Y_jk['C']['D'] + Y_jk['C']['E']
        modelo += 0.060 * X_ij['A']['D'] + 0.080 * X_ij['B']['D'] + 0.062 * X_ij['C']['D'] == Y_jk['D']['D'] + Y_jk['D']['E']
        modelo += 0.060 * X_ij['A']['E'] + 0.080 * X_ij['B']['E'] + 0.062 * X_ij['C']['E'] == Y_jk['E']['D'] + Y_jk['E']['E']

        # Resolver el modelo
        modelo.solve()

        # Obtener resultados
        costo_total = value(modelo.objective)
        
        # Ver qué plantas se abren
        for j in ['B','C','D','E']:
            if value(W_j[j]) == 1:
                plantas_abiertas.append(f"Planta {j}")
    
    return render_template("home.html", 
                         costo_total=costo_total,
                         plantas_abiertas=plantas_abiertas)

if __name__ == "__main__":
    app.run(debug=True)