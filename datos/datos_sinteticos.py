"""
Módulo generador de datos sintéticos para el manual completo.
Provee datos de Ventas, Compras e Inventarios con sesgo realista.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ========== CONSTANTES ==========
PRODUCTOS = [
    ("LAP001", "Laptop Pro 15", "Electrónica", 12000, 15000),
    ("LAP002", "Laptop Air 13", "Electrónica", 9000, 11500),
    ("MON001", "Monitor 27 4K", "Electrónica", 5500, 7200),
    ("MON002", "Monitor 24 HD", "Electrónica", 2500, 3400),
    ("TEC001", "Teclado Mecánico", "Periféricos", 800, 1400),
    ("TEC002", "Teclado Inalámbrico", "Periféricos", 450, 750),
    ("MOU001", "Mouse Ergonómico", "Periféricos", 350, 650),
    ("MOU002", "Mouse Gamer", "Periféricos", 600, 1100),
    ("AUD001", "Audífonos Bluetooth", "Audio", 1200, 2200),
    ("AUD002", "Parlante Portátil", "Audio", 800, 1500),
    ("AUD003", "Micrófono USB", "Audio", 1500, 2800),
    ("WEB001", "Webcam HD", "Cámaras", 900, 1700),
    ("WEB002", "Cámara Seguridad", "Cámaras", 1800, 3200),
    ("DIS001", "SSD 1TB", "Almacenamiento", 1500, 2500),
    ("DIS002", "HDD 4TB", "Almacenamiento", 1200, 1900),
    ("DIS003", "USB 64GB", "Almacenamiento", 150, 350),
    ("RED001", "Router WiFi 6", "Redes", 1200, 2100),
    ("RED002", "Switch 8 puertos", "Redes", 600, 1100),
    ("SOF001", "Office 365 1 año", "Software", 900, 1800),
    ("SOF002", "Antivirus 3 equipos", "Software", 400, 800),
    ("PAP001", "Papel Bond 5000 hojas", "Papelería", 200, 400),
    ("PAP002", "Tinta Impresora", "Papelería", 250, 500),
    ("MUE001", "Silla Ergonómica", "Muebles", 3500, 6500),
    ("MUE002", "Escritorio Eléctrico", "Muebles", 4500, 8500),
    ("MUE003", "Lámpara LED", "Muebles", 300, 600),
]

PROVEEDORES = [
    ("PROV001", "Distribuidora Tecnológica S.A.", 85, 3),
    ("PROV002", "Importaciones Globales Ltda.", 70, 7),
    ("PROV003", "Suministros Empresariales C.A.", 92, 2),
    ("PROV004", "TecnoPartes del Sur", 60, 10),
    ("PROV005", "Logística Integral de Cómputo", 88, 4),
    ("PROV006", "Comercializadora Digital Express", 75, 5),
    ("PROV007", "Mayorista de Tecnología", 95, 1),
]

SUCURSALES = [
    "Matriz CDMX", "Sucursal Monterrey", "Sucursal Guadalajara",
    "Sucursal Puebla", "Sucursal Querétaro", "Sucursal Cancún",
    "Sucursal Toluca", "Sucursal Mérida", "Sucursal Tijuana"
]

CLIENTES = [
    "Cliente Corp", "Empresa XYZ", "Comercial MX", "Distribuidora ABC",
    "Tienda 123", "Mayorista JJ", "Comprador Final", "Empresa Beta",
    "Soluciones Inc", "Grupo Gamma", "Venta Mostrador", "Cliente Premium"
]


def generar_ventas(n_dias=180, productos=None, sucursales=None, clientes=None):
    """Genera DataFrame de ventas diarias."""
    if productos is None:
        productos = PRODUCTOS
    if sucursales is None:
        sucursales = SUCURSALES
    if clientes is None:
        clientes = CLIENTES

    registros = []
    inicio = datetime(2024, 1, 1)

    for dia in range(n_dias):
        fecha = inicio + timedelta(days=dia)
        dia_semana = fecha.weekday()
        mes = fecha.month

        # Estacionalidad: más ventas en fines de mes y ciertos meses
        factor_fin_mes = 1.3 if dia % 30 >= 27 else 1.0
        factor_semana = 0.7 if dia_semana >= 5 else 1.0
        factor_mes = 1.5 if mes in [11, 12] else (1.2 if mes in [6, 7] else 1.0)

        for producto in productos:
            sku, nombre, categoria, costo, precio = producto
            # Probabilidad de venta por producto
            prob_venta = np.random.beta(2, 5)
            if np.random.random() > prob_venta:
                continue

            cantidad = int(np.random.poisson(
                max(1, 10 * factor_fin_mes * factor_semana * factor_mes)
            ) * np.random.uniform(0.3, 1.5))
            cantidad = max(1, min(cantidad, 50))

            descuento = np.random.choice([0, 0.05, 0.10, 0.15, 0.20],
                                        p=[0.5, 0.25, 0.15, 0.07, 0.03])
            precio_final = round(precio * (1 - descuento), 2)
            costo_total = round(costo * cantidad, 2)
            ingreso = round(precio_final * cantidad, 2)

            registros.append({
                "fecha": fecha,
                "sku": sku,
                "producto": nombre,
                "categoria": categoria,
                "sucursal": np.random.choice(sucursales),
                "cliente": np.random.choice(clientes),
                "cantidad": cantidad,
                "precio_unitario": precio_final,
                "costo_unitario": costo,
                "ingreso": ingreso,
                "costo_total": costo_total,
                "margen": round(ingreso - costo_total, 2),
                "margen_pct": round((ingreso / costo_total - 1) * 100, 1),
                "descuento": descuento,
                "dia_semana": dia_semana,
                "mes": mes,
            })

    df = pd.DataFrame(registros)
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


def generar_inventario(productos=None):
    """Genera DataFrame del inventario actual."""
    if productos is None:
        productos = PRODUCTOS

    inventario = []
    for producto in productos:
        sku, nombre, categoria, costo, precio = producto
        stock_actual = np.random.randint(0, 200)
        stock_minimo = np.random.randint(5, 30)
        stock_maximo = np.random.randint(50, 300)
        demanda_diaria = np.random.poisson(max(1, int(np.random.exponential(3))))

        inventario.append({
            "sku": sku,
            "producto": nombre,
            "categoria": categoria,
            "costo": costo,
            "precio": precio,
            "stock_actual": stock_actual,
            "stock_minimo": stock_minimo,
            "stock_maximo": stock_maximo,
            "demanda_diaria_prom": demanda_diaria,
            "dias_para_agotar": round(stock_actual / max(demanda_diaria, 1), 1),
            "valor_inventario": round(stock_actual * costo, 2),
            "necesita_reposicion": stock_actual < stock_minimo,
        })

    return pd.DataFrame(inventario)


def generar_compras(n_ordenes=200, productos=None, proveedores=None):
    """Genera DataFrame de órdenes de compra."""
    if productos is None:
        productos = PRODUCTOS
    if proveedores is None:
        proveedores = PROVEEDORES

    ordenes = []
    inicio = datetime(2024, 1, 1)

    for _ in range(n_ordenes):
        producto = productos[np.random.randint(0, len(productos))]
        proveedor = proveedores[np.random.randint(0, len(proveedores))]
        sku, nombre, categoria, costo, precio = producto
        prov_id, prov_nombre, calidad, plazo = proveedor

        cantidad = int(np.random.gamma(3, 10)) + 1
        costo_unitario = round(costo * np.random.uniform(0.85, 1.15), 2)
        dias_entrega = int(max(1, np.random.poisson(plazo)))
        fecha_orden = inicio + timedelta(days=np.random.randint(0, 365))
        fecha_entrega = fecha_orden + timedelta(days=dias_entrega)
        entregado = np.random.random() > 0.15
        dias_retraso = max(0, int(np.random.normal(0, 3))) if entregado else None

        ordenes.append({
            "orden_id": f"OC-{len(ordenes)+1:05d}",
            "fecha_orden": fecha_orden,
            "fecha_entrega": fecha_entrega if entregado else None,
            "proveedor_id": prov_id,
            "proveedor": prov_nombre,
            "calidad_proveedor": calidad,
            "sku": sku,
            "producto": nombre,
            "categoria": categoria,
            "cantidad": cantidad,
            "costo_unitario": costo_unitario,
            "costo_total": round(cantidad * costo_unitario, 2),
            "dias_estimados": plazo,
            "dias_reales": dias_entrega if entregado else None,
            "retraso": dias_retraso,
            "entregado": entregado,
            "puntual": dias_retraso is not None and dias_retraso <= 0,
        })

    return pd.DataFrame(ordenes)


def generar_clientes_con_compras(n_clientes=200):
    """Genera DataFrame de clientes con historial de compras."""
    clientes_data = []
    for i in range(n_clientes):
        cliente_id = f"C{i+1:05d}"
        recencia = np.random.randint(1, 365)
        frecuencia = int(np.random.exponential(5)) + 1
        monto = round(np.random.gamma(5, 200) + 100, 2)
        clientes_data.append({
            "cliente_id": cliente_id,
            "recencia_dias": recencia,
            "frecuencia_compras": frecuencia,
            "monto_total": monto,
            "ticket_promedio": round(monto / max(frecuencia, 1), 2),
            "antiguedad_dias": np.random.randint(30, 2000),
        })

    df = pd.DataFrame(clientes_data)
    df["segmento_rfm"] = pd.qcut(df["recencia_dias"], 4, labels=["Alto", "Medio-Alto", "Medio-Bajo", "Bajo"])
    return df


def generar_resenas(n_resenas=100):
    """Genera reseñas de productos."""
    reseñas = []
    productos = ["Laptop", "Monitor", "Teclado", "Mouse", "Audífonos", "Webcam", "SSD", "Router"]
    textos_pos = [
        "Excelente producto, muy recomendable",
        "Buena calidad y rápido envío",
        "Cumple con lo prometido, estoy satisfecho",
        "Mejor de lo que esperaba, volvería a comprar",
        "Muy buen rendimiento a este precio",
    ]
    textos_neg = [
        "No funciona como esperaba, defectuoso",
        "Mala calidad, se rompió en una semana",
        "No cumple con las especificaciones",
        "Pésimo servicio, llegó dañado",
        "Sobrevalorado, no vale lo que cuesta",
    ]
    textos_neut = [
        "Cumple su función básica",
        "Está bien para el precio que tiene",
        "Podría ser mejor pero es aceptable",
        "Entrega a tiempo, el producto es normal",
        "Es lo que esperaba, ni más ni menos",
    ]

    for i in range(n_resenas):
        producto = np.random.choice(productos)
        sentimiento = np.random.choice(["positivo", "negativo", "neutro"], p=[0.6, 0.15, 0.25])
        if sentimiento == "positivo":
            texto = np.random.choice(textos_pos)
            puntuacion = np.random.randint(4, 6)
        elif sentimiento == "negativo":
            texto = np.random.choice(textos_neg)
            puntuacion = np.random.randint(1, 3)
        else:
            texto = np.random.choice(textos_neut)
            puntuacion = 3
        reseñas.append({
            "reseña_id": f"R{i+1:05d}",
            "producto": producto,
            "cliente": np.random.choice(CLIENTES),
            "texto": texto,
            "puntuacion": puntuacion,
            "sentimiento": sentimiento,
            "fecha": datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365)),
        })
    return pd.DataFrame(reseñas)


if __name__ == "__main__":
    print("Generando datos sintéticos...")
    ventas = generar_ventas(180)
    inventario = generar_inventario()
    compras = generar_compras(200)
    clientes = generar_clientes_con_compras(200)
    resenas = generar_resenas(100)

    ventas.to_csv("datos/ventas.csv", index=False)
    inventario.to_csv("datos/inventario.csv", index=False)
    compras.to_csv("datos/compras.csv", index=False)
    clientes.to_csv("datos/clientes.csv", index=False)
    resenas.to_csv("datos/resenas.csv", index=False)

    print(f"Ventas: {len(ventas)} registros")
    print(f"Inventario: {len(inventario)} productos")
    print(f"Compras: {len(compras)} órdenes")
    print(f"Clientes: {len(clientes)} clientes")
    print(f"Reseñas: {len(resenas)} reseñas")
    print("¡Datos generados exitosamente!")
