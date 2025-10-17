from flask import Flask, render_template, request
from pulp import *

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def optimizar_bauxita():
    # Variables para resultados
    costo_total = None
    plantas_abiertas = []
    
    if request.method == "POST":
        # Obtener los valores binarios ingresados por el usuario
        W_B_input = request.form.get("W_B")
        W_C_input = request.form.get("W_C")
        W_D_input = request.form.get("W_D")
        W_E_input = request.form.get("W_E")
        
        print("Valores binarios ingresados:")
        print("W_B:", W_B_input)
        print("W_C:", W_C_input) 
        print("W_D:", W_D_input)
        print("W_E:", W_E_input)
        
        # EJECUTAR EL MODELO DE OPTIMIZACIÓN
        modelo = LpProblem("La_bauxita", LpMinimize)
        
        # Variables de decisión
        X_ij = LpVariable.dicts("X", (['A','B','C'], ['B','C','D','E']), lowBound=0)
        Y_jk = LpVariable.dicts("Y", (['B','C','D','E'], ['D','E']), lowBound=0)
        W_j = LpVariable.dicts("W", ['B','C','D','E'], cat='Binary')
        
        # Función objetivo
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
        
        Costo_fijo_plantas = (
            3000000 * W_j['B'] + 
            2500000 * W_j['C'] + 
            4800000 * W_j['D'] + 
            6000000 * W_j['E']
        )

        modelo += (
            Costo_explotacion + 
            Costo_produccion + 
            Costo_procesamiento + 
            Costo_transporte_bauxita + 
            Costo_transporte_alumina + 
            Costo_fijo_plantas
        )

        # Restricciones normales
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

        # ASIGNAR VALORES BINARIOS INGRESADOS POR EL USUARIO
        if W_B_input is not None and W_B_input != '':
            modelo += W_j['B'] == int(W_B_input)
        if W_C_input is not None and W_C_input != '':
            modelo += W_j['C'] == int(W_C_input)
        if W_D_input is not None and W_D_input != '':
            modelo += W_j['D'] == int(W_D_input)
        if W_E_input is not None and W_E_input != '':
            modelo += W_j['E'] == int(W_E_input)

        # Resolver el modelo
        modelo.solve()

        # Obtener resultados
        costo_total = value(modelo.objective)
        
        # Recopilar información de plantas
        for j in ['B','C','D','E']:
            estado = "ABIERTA" if value(W_j[j]) == 1 else "CERRADA"
            plantas_abiertas.append(f"Planta {j}: {estado}")
    
    return render_template("home.html", 
                         costo_total=costo_total,
                         plantas_abiertas=plantas_abiertas)

if __name__ == "__main__":
    app.run(debug=True)