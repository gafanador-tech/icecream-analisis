# Análisis estadístico — Ice Cream × Emotion

Programa que corre en una sola ejecución todos los análisis del trabajo de grado a partir de la exportación de la plataforma (protocolo 1.2.0).

## Uso

```
pip install -r requirements.txt
python run_analysis.py <carpeta_de_la_exportacion> <carpeta_de_resultados>
```

La carpeta de la exportación es el ZIP descargado del panel de investigador, ya descomprimido. Los datos de los participantes **no** forman parte de este repositorio.

## Qué hace

1. Incluye solo las sesiones reales y completas del protocolo 1.2.0 y verifica su integridad (15 emparejamientos, 60 y 100 escalas por sesión).
2. Perfil de la muestra y tablas de elecciones por sabor y familia.
3. **H1:** chi-cuadrado de bondad de ajuste por sabor (corrección de Holm) y GEE binomial de la elección de un emoticón positivo según la familia de sabor.
4. **H2:** GEE del tiempo de respuesta, la desviación máxima, el área bajo la curva y los cambios de dirección según el acuerdo entre participantes (dejando fuera a cada participante). Se excluyen los ensayos con tiempos superiores a la media más dos desviaciones estándar del logaritmo del tiempo (Salgado-Montejo et al., 2015a).
5. **H3 y H4:** logit condicional sobre los cinco emoticones de cada ensayo, con errores estándar robustos agrupados por participante: modelo base, más la distancia entre perfiles de sabor (Fase 3), más las diferencias afectivas (Fase 2) y ambos.
6. Análisis de sensibilidad: participantes que evaluaron antes del emparejamiento y muestra sin sesiones en inglés.

Cada tabla incluye el valor *p* exacto y su versión en formato APA. `registro_analisis.txt` resume la ejecución, las versiones de las librerías y los controles de integridad.
