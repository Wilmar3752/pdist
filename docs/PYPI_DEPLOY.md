# 🚀 Deploy Automático a PyPI - Guía Completa

## 📋 Índice
1. [Cómo Funciona](#cómo-funciona)
2. [Configuración Inicial](#configuración-inicial)
3. [Proceso de Release](#proceso-de-release)
4. [Trusted Publishing](#trusted-publishing)
5. [Troubleshooting](#troubleshooting)

---

## 🔍 Cómo Funciona

Tu proyecto usa **GitHub Actions** con **Trusted Publishing** (PyPI) para deploy automático.

### Flujo General

```
1. Creas un Release en GitHub
         ↓
2. GitHub Actions se activa automáticamente
         ↓
3. Job "release-build": Construye el paquete
         ↓
4. Job "pypi-publish": Publica a PyPI
         ↓
5. Tu paquete está disponible en PyPI! 🎉
```

### Workflow Actual (`.github/workflows/python-publish.yml`)

```yaml
name: Upload Python Package

on:
  release:
    types: [published]  # ← Se activa cuando publicas un Release
```

**Trigger**: El workflow se ejecuta cuando creas un **Release** en GitHub (no con tags o push normales)

---

## 🏗️ Anatomía del Workflow

### Job 1: `release-build` (Construir el paquete)

```yaml
release-build:
  runs-on: ubuntu-latest
  
  steps:
    - uses: actions/checkout@v4           # Descarga tu código
    
    - uses: actions/setup-python@v5       # Instala Python 3.x
      with:
        python-version: "3.x"
    
    - name: Build release distributions   # Construye el paquete
      run: |
        python -m pip install build
        python -m build                   # Crea dist/*.whl y dist/*.tar.gz
    
    - name: Upload distributions          # Guarda los archivos generados
      uses: actions/upload-artifact@v4
      with:
        name: release-dists
        path: dist/
```

**¿Qué hace?**
- ✅ Descarga tu código desde GitHub
- ✅ Instala Python
- ✅ Ejecuta `python -m build` que genera:
  - `dist/pdist-0.1.0-py3-none-any.whl` (wheel)
  - `dist/pdist-0.1.0.tar.gz` (source distribution)
- ✅ Guarda estos archivos como "artifacts"

### Job 2: `pypi-publish` (Publicar a PyPI)

```yaml
pypi-publish:
  runs-on: ubuntu-latest
  needs:
    - release-build                       # Espera a que termine release-build
  
  permissions:
    id-token: write                       # CRÍTICO para Trusted Publishing
  
  environment:
    name: pypi                            # Usa el environment "pypi"
  
  steps:
    - name: Retrieve release distributions
      uses: actions/download-artifact@v4  # Descarga los .whl y .tar.gz
      with:
        name: release-dists
        path: dist/
    
    - name: Publish release distributions to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1  # Publica a PyPI
      with:
        packages-dir: dist/
```

**¿Qué hace?**
- ✅ Descarga los archivos generados en Job 1
- ✅ Los publica a PyPI usando **Trusted Publishing** (sin passwords!)
- ✅ Tu paquete queda disponible en `https://pypi.org/project/pdist`

---

## 🔐 Configuración Inicial (Trusted Publishing)

### ¿Qué es Trusted Publishing?

**Antes (método antiguo)**:
- Necesitabas generar un API token en PyPI
- Guardarlo como secreto en GitHub
- Riesgo de seguridad si el token se filtra

**Ahora (Trusted Publishing)**:
- ✅ Sin tokens, sin passwords
- ✅ PyPI confía directamente en GitHub Actions
- ✅ Más seguro y más simple

### Pasos para Configurar Trusted Publishing

#### 1️⃣ En PyPI (https://pypi.org)

1. **Crea una cuenta** en PyPI (si no tienes)
   - Ve a: https://pypi.org/account/register/

2. **Crea el proyecto "pdist"** (solo la primera vez)
   - Ve a: https://pypi.org/manage/account/publishing/
   - Click en "Add a new pending publisher"
   
3. **Completa el formulario**:
   ```
   PyPI Project Name: pdist
   Owner: <tu-usuario-github>
   Repository name: pdist
   Workflow name: python-publish.yml
   Environment name: pypi
   ```

4. **Guarda** - PyPI ahora espera publicaciones desde tu GitHub Actions

#### 2️⃣ En GitHub (tu repositorio)

1. **Crea un Environment**:
   - Ve a: Settings → Environments → New environment
   - Nombre: `pypi` (debe coincidir con el workflow)

2. **(Opcional) Agrega protecciones**:
   - Required reviewers: Requiere aprobación manual antes de publicar
   - Wait timer: Espera X minutos antes de publicar
   - Deployment branches: Solo desde `main` branch

3. **Verifica el workflow**:
   - El archivo `.github/workflows/python-publish.yml` debe existir (✅ ya lo tienes)

---

## 🚀 Proceso de Release (Cómo publicar a PyPI)

### Paso a Paso

#### 1️⃣ Actualiza la versión

Edita `pyproject.toml`:

```toml
[project]
name = "pdist"
version = "0.1.0"  # ← Cambia esto (0.1.1, 0.2.0, 1.0.0, etc.)
```

**Versionado Semántico**:
- `0.1.0 → 0.1.1`: Bug fixes (patch)
- `0.1.0 → 0.2.0`: Nuevas features compatibles (minor)
- `0.1.0 → 1.0.0`: Cambios incompatibles (major)

#### 2️⃣ Actualiza el CHANGELOG

```markdown
# CHANGELOG.md

## [0.1.1] - 2024-01-06

### Added
- Supresión automática de warnings en DistributionFitter
- Documentación completa sobre warnings

### Fixed
- Corrección en parámetros de distribución Beta

### Changed
- Mejora en la presentación de resultados
```

#### 3️⃣ Commit y Push

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.1.1"
git push origin main
```

#### 4️⃣ Crea un Tag (opcional pero recomendado)

```bash
git tag -a v0.1.1 -m "Release version 0.1.1"
git push origin v0.1.1
```

#### 5️⃣ Crea un Release en GitHub

**Opción A: Via Web UI**

1. Ve a tu repo en GitHub
2. Click en "Releases" → "Draft a new release"
3. Completa el formulario:
   ```
   Tag: v0.1.1 (o crea uno nuevo)
   Release title: v0.1.1
   Description: 
   ## What's New
   - Automatic warning suppression
   - Improved documentation
   - Bug fixes
   
   See full CHANGELOG.md for details
   ```
4. Click "Publish release" → **Esto activa el workflow!** 🎉

**Opción B: Via GitHub CLI**

```bash
gh release create v0.1.1 \
  --title "v0.1.1" \
  --notes "See CHANGELOG.md for details"
```

#### 6️⃣ Espera a que GitHub Actions termine

1. Ve a: Actions → Upload Python Package
2. Verás los dos jobs:
   - ✅ release-build (construye el paquete)
   - ✅ pypi-publish (publica a PyPI)

3. Si hay errores, los verás aquí

#### 7️⃣ Verifica en PyPI

Después de ~2-3 minutos:

```bash
# Busca tu paquete
https://pypi.org/project/pdist

# Instálalo para probar
pip install pdist

# O actualiza
pip install --upgrade pdist
```

---

## 📊 Diagrama del Flujo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│  DESARROLLADOR                                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Actualiza version en pyproject.toml                         │
│  2. Commit & Push a main                                        │
│  3. Crea Release en GitHub                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  GITHUB ACTIONS (Automático)                                    │
│                                                                  │
│  Job 1: release-build                                           │
│    ✓ Checkout código                                            │
│    ✓ Instalar Python                                            │
│    ✓ Ejecutar: python -m build                                  │
│    ✓ Generar: dist/*.whl y dist/*.tar.gz                        │
│    ✓ Guardar artifacts                                          │
│                                                                  │
│  Job 2: pypi-publish (necesita Job 1)                           │
│    ✓ Descargar artifacts                                        │
│    ✓ Autenticar con PyPI (Trusted Publishing)                   │
│    ✓ Publicar a PyPI                                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  PYPI                                                            │
│                                                                  │
│  ✓ Paquete publicado en https://pypi.org/project/pdist          │
│  ✓ Usuarios pueden instalar: pip install pdist                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Comandos Útiles

### Construir localmente (antes de release)

```bash
# Instalar herramientas de build
pip install build twine

# Construir el paquete
python -m build

# Verificar los archivos generados
ls -lh dist/

# Verificar que el paquete está bien formado
twine check dist/*

# (Opcional) Probar instalación local
pip install dist/pdist-0.1.0-py3-none-any.whl
```

### Publicar manualmente (para testing)

```bash
# A TestPyPI (para probar)
twine upload --repository testpypi dist/*

# A PyPI (producción)
twine upload dist/*
```

⚠️ **Nota**: Con Trusted Publishing, NO necesitas hacer esto manualmente. GitHub Actions lo hace automáticamente.

---

## 🐛 Troubleshooting

### Problema 1: "Trusted publishing is not configured"

**Error**: 
```
Error: Trusted publishing exchange failure
```

**Solución**:
1. Ve a PyPI → Manage Account → Publishing
2. Verifica que el "pending publisher" esté configurado correctamente
3. Asegúrate que los nombres coincidan:
   - Repository: `usuario/pdist`
   - Workflow: `python-publish.yml`
   - Environment: `pypi`

### Problema 2: "Version already exists"

**Error**:
```
HTTPError: 400 Bad Request
File already exists
```

**Solución**:
- No puedes sobrescribir una versión en PyPI
- Incrementa la versión en `pyproject.toml`
- Ejemplo: `0.1.0` → `0.1.1`

### Problema 3: Workflow no se activa

**Síntomas**: Publicas un release pero Actions no se ejecuta

**Solución**:
1. Verifica que el archivo esté en `.github/workflows/python-publish.yml`
2. Asegúrate de crear un **Release**, no solo un tag
3. Revisa Actions → "Upload Python Package" para ver logs

### Problema 4: Build falla

**Error común**:
```
ModuleNotFoundError: No module named 'setuptools'
```

**Solución**:
- Ya está resuelto en tu workflow (línea 32 instala `build`)
- Si persiste, verifica `pyproject.toml`

### Problema 5: Environment "pypi" no existe

**Error**:
```
Environment protection rules not satisfied
```

**Solución**:
1. Ve a Settings → Environments
2. Crea environment llamado `pypi`
3. (Opcional) Configura protecciones

---

## 📝 Checklist Antes del Primer Release

- [ ] Cuenta creada en PyPI
- [ ] Trusted Publishing configurado en PyPI
- [ ] Environment "pypi" creado en GitHub
- [ ] Workflow `.github/workflows/python-publish.yml` existe
- [ ] `pyproject.toml` tiene metadata correcta (name, version, description)
- [ ] README.md está completo
- [ ] LICENSE file existe
- [ ] Tests pasan (`pytest`)
- [ ] Código formateado (`black`, `isort`)
- [ ] CHANGELOG.md actualizado

---

## 🎯 Resumen Ejecutivo

### Para tu primer release:

1. **Configura Trusted Publishing en PyPI** (una sola vez)
2. **Actualiza version** en `pyproject.toml`
3. **Crea un Release en GitHub**
4. **Espera** ~2-3 minutos
5. **¡Listo!** Tu paquete está en PyPI

### Para releases futuros:

Solo necesitas repetir pasos 2-4. ¡Es así de simple!

```bash
# El proceso completo en 4 comandos:
vim pyproject.toml           # Cambiar version
git commit -am "bump v0.2.0"
git push
gh release create v0.2.0     # GitHub Actions hace el resto!
```

---

## 📚 Referencias

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)

---

**¿Necesitas ayuda?** Abre un issue en el repositorio o consulta la documentación oficial de PyPI.

