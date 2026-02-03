from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from urllib.parse import urlparse


st.set_page_config(page_title="AutoML Hackathon", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --primary: #6d5dfc;
        --primary-dark: #5b4ae3;
        --secondary: #16c79a;
        --accent: #f4b740;
        --bg: #f8f9fb;
        --text: #101828;
        --muted: #667085;
        --card: #ffffff;
        --border: #eaecf0;
        --shadow: 0px 12px 30px rgba(16, 24, 40, 0.08);
    }

    .stApp {
        background: var(--bg);
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #f9fafb !important;
    }

    .hero {
        background: linear-gradient(135deg, rgba(109, 93, 252, 0.12), rgba(22, 199, 154, 0.12));
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 28px 32px;
        box-shadow: var(--shadow);
    }

    .hero h1 {
        font-size: 2.4rem;
        margin-bottom: 0.4rem;
        color: var(--text);
    }

    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: var(--shadow);
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(109, 93, 252, 0.12);
        color: var(--primary-dark);
        font-weight: 600;
        font-size: 0.85rem;
    }

    .metric-row {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        margin-top: 12px;
    }

    .metric {
        background: #f2f4f7;
        border-radius: 16px;
        padding: 14px 16px;
        min-width: 160px;
    }

    .metric h4 {
        margin: 0 0 4px 0;
        font-size: 0.85rem;
        color: var(--muted);
        font-weight: 600;
    }

    .metric span {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text);
    }

    .stButton > button {
        border-radius: 999px;
        padding: 0.6rem 1.4rem;
        background: var(--primary);
        color: white;
        border: none;
        font-weight: 600;
        box-shadow: 0px 10px 22px rgba(109, 93, 252, 0.25);
    }

    .stButton > button:hover {
        background: var(--primary-dark);
        color: white;
    }

    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        border-radius: 12px !important;
    }

    .stDataFrame, .stDataEditor {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    .section-title {
        font-weight: 700;
        font-size: 1.2rem;
        color: var(--text);
        margin-bottom: 0.6rem;
    }

    .hint {
        color: var(--muted);
        font-size: 0.92rem;
    }

    .transform-title {
        font-weight: 700;
        color: var(--text);
        margin-top: 0.6rem;
    }

    .transform-caption {
        color: var(--muted);
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
    }

    .prediction-panel {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    .prediction-result {
        background: linear-gradient(135deg, rgba(109, 93, 252, 0.12), rgba(244, 183, 64, 0.12));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px 20px;
    }

    .prediction-result h3 {
        margin: 0 0 6px 0;
    }

    .prediction-meta {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")


st.markdown(
    """
    <div class="hero">
        <div class="pill">AutoML Experience</div>
        <h1>Plataforma AutoML</h1>
        <p>Crie pipelines modernos de machine learning com um fluxo visual bonito, rápido e amigável. Do upload ao deploy em poucos cliques.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> Any:
    response = requests.get(f"{API_URL}{path}")
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: Dict[str, Any] | None = None) -> Any:
    response = requests.post(f"{API_URL}{path}", json=payload)
    response.raise_for_status()
    return response.json()


def api_post_file(path: str, file_name: str, file_bytes: bytes) -> Any:
    response = requests.post(f"{API_URL}{path}", files={"file": (file_name, file_bytes)})
    response.raise_for_status()
    return response.json()


def api_get_file(path: str) -> bytes:
    response = requests.get(f"{API_URL}{path}")
    response.raise_for_status()
    return response.content


def get_deployment_from_path() -> str | None:
    ctx = get_script_run_ctx()
    path = ""
    if ctx and getattr(ctx, "request", None):
        path = ctx.request.path or ""
        if path in {"", "/"}:
            referer = ctx.request.headers.get("referer")
            if referer:
                parsed = urlparse(referer)
                path = parsed.path
    if path and path != "/":
        return path.lstrip("/")
    query_params = st.experimental_get_query_params()
    if query_params.get("deploy"):
        return query_params["deploy"][0]
    return None


def render_deployment_page(deployment_name: str) -> None:
    st.markdown("<div class='section-title'>Predições do deploy</div>", unsafe_allow_html=True)
    try:
        deployment = api_get(f"/deployments/{deployment_name}")
    except requests.HTTPError:
        st.error("Deploy não encontrado. Volte ao painel principal para criar um.")
        return

    st.markdown(
        f"""
        <div class="card">
            <h3>Deploy: {deployment_name}</h3>
            <p class="hint">Envie um CSV/Excel e receba uma planilha com a coluna <strong>prediction</strong>. Baixe direto daqui.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Detalhes do deploy", expanded=False):
        st.json(deployment)

    col_upload, col_tips = st.columns([1.1, 0.9])
    with col_upload:
        st.markdown("<div class='prediction-panel'>", unsafe_allow_html=True)
        upload = st.file_uploader("Arquivo para predição", type=["csv", "xlsx", "xls"])
        st.caption("Inclua as mesmas colunas usadas no treino. A coluna target é ignorada automaticamente.")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_tips:
        st.markdown(
            """
            <div class="card">
                <h3>Checklist rápido</h3>
                <ul>
                    <li>Mesma ordem de colunas do treino.</li>
                    <li>Sem valores nulos onde o modelo não aceita.</li>
                    <li>CSV separado por vírgula.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if upload is not None and st.button("Executar predição"):
        try:
            result = api_post_file(f"/deployments/{deployment_name}/predict-file", upload.name, upload.getvalue())
        except requests.HTTPError as exc:
            st.error(str(exc))
            return
        st.success("Predição concluída")
        st.markdown(
            f"""
            <div class="prediction-result">
                <h3>Arquivo pronto para download</h3>
                <div class="prediction-meta">
                    <span>Linhas previstas: <strong>{result['rows']}</strong></span>
                    <span>Arquivo: <strong>{result['output_filename']}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        output_name = result.get("output_filename") or Path(result["output_file"]).name
        download_path = result.get("download_url") or f"/files/{output_name}"
        file_bytes = api_get_file(download_path)
        st.download_button(
            label="Baixar predição",
            data=file_bytes,
            file_name=output_name,
            mime="text/csv",
        )


deployment_from_path = get_deployment_from_path()
if deployment_from_path:
    render_deployment_page(deployment_from_path)
    st.stop()


section = st.sidebar.radio(
    "Fluxo",
    ["Upload", "Explorar", "Transformar", "Treinar", "Modelos", "Predições"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Centro de controle")
st.sidebar.caption("API FastAPI + Frontend Streamlit")


if section == "Upload":
    st.markdown("<div class='section-title'>1. Upload de dataset</div>", unsafe_allow_html=True)
    col_info, col_upload = st.columns([1.2, 1])
    with col_info:
        st.markdown(
            """
            <div class="card">
                <h3>Importe dados em segundos</h3>
                <p class="hint">Envie arquivos CSV ou Excel e tenha uma visão geral instantânea do seu dataset, com métricas e perfis automáticos.</p>
                <div class="metric-row">
                    <div class="metric"><h4>Tipos aceitos</h4><span>CSV, XLSX</span></div>
                    <div class="metric"><h4>Processamento</h4><span>Instantâneo</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_upload:
        upload = st.file_uploader("Escolha um CSV ou Excel", type=["csv", "xlsx", "xls"])
        if upload is not None:
            files = {"file": (upload.name, upload.getvalue())}
            response = requests.post(f"{API_URL}/datasets/upload", files=files)
            if response.ok:
                st.success("Dataset carregado!")
                st.json(response.json())
            else:
                st.error(response.text)

if section == "Explorar":
    st.markdown("<div class='section-title'>2. Exploração dos dados</div>", unsafe_allow_html=True)
    datasets = api_get("/datasets")
    dataset_map = {d["filename"] + f" ({d['dataset_id'][:8]})": d for d in datasets}
    if dataset_map:
        selection = st.selectbox("Escolha o dataset", list(dataset_map.keys()))
        dataset_id = dataset_map[selection]["dataset_id"]
        summary = api_get(f"/datasets/{dataset_id}/summary")
        sample_df = pd.DataFrame(summary["sample"])

        overview, quality = st.columns([1.3, 1])
        with overview:
            st.markdown(
                """
                <div class="card">
                    <h3>Visão geral</h3>
                    <p class="hint">Resumo imediato da estrutura do dataset para orientar decisões rápidas.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(sample_df, use_container_width=True)
        with quality:
            st.markdown(
                """
                <div class="card">
                    <h3>Qualidade dos dados</h3>
                    <p class="hint">Tipos, ausências e estatísticas em um único local.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.json({"dtypes": summary["dtypes"], "missing": summary["missing"]})
            st.json(summary["summary"])

        st.markdown("<div class='section-title'>Correlação numérica</div>", unsafe_allow_html=True)
        st.json(summary["correlations"])
    else:
        st.info("Faça upload de um dataset primeiro.")

if section == "Transformar":
    st.markdown("<div class='section-title'>3. Transformações & edição manual</div>", unsafe_allow_html=True)
    datasets = api_get("/datasets")
    dataset_map = {d["filename"] + f" ({d['dataset_id'][:8]})": d for d in datasets}
    if dataset_map:
        selection = st.selectbox("Escolha o dataset", list(dataset_map.keys()))
        dataset_id = dataset_map[selection]["dataset_id"]
        preview = api_get(f"/datasets/{dataset_id}/preview")
        df_preview = pd.DataFrame(preview)

        st.markdown(
            """
            <div class="card">
                <h3>Edição inteligente</h3>
                <p class="hint">Edite a planilha manualmente, arraste colunas e faça ajustes finos antes das transformações. Depois, você pode exportar o CSV atualizado.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        edited_df = st.data_editor(
            df_preview,
            use_container_width=True,
            num_rows="dynamic",
            key="editor",
        )

        col_export, col_hint = st.columns([0.4, 0.6])
        with col_export:
            st.download_button(
                label="Baixar CSV editado",
                data=edited_df.to_csv(index=False).encode("utf-8"),
                file_name="dataset_editado.csv",
                mime="text/csv",
            )
        with col_hint:
            st.info("Dica: após baixar, reenvie o CSV para criar uma nova versão do dataset.")

        st.markdown("<div class='section-title'>Transformações rápidas</div>", unsafe_allow_html=True)
        st.markdown("<div class='transform-title'>1. Remover colunas</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='transform-caption'>Selecione as colunas que devem ser removidas do dataset.</div>",
            unsafe_allow_html=True,
        )
        drop_columns = st.multiselect("Remover colunas", df_preview.columns.tolist())

        st.markdown("<div class='transform-title'>2. Preencher valores nulos</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='transform-caption'>Escolha a coluna e o valor que será usado para preencher valores ausentes.</div>",
            unsafe_allow_html=True,
        )
        fill_column = st.selectbox("Coluna para preencher nulos", [""] + df_preview.columns.tolist())
        fill_value = st.text_input("Valor para preencher")

        st.markdown("<div class='transform-title'>3. Criar nova coluna</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='transform-caption'>Defina um nome e uma expressão pandas para gerar uma coluna derivada.</div>",
            unsafe_allow_html=True,
        )
        add_name = st.text_input("Nova coluna (nome)")
        add_expr = st.text_input("Nova coluna (expressão pandas, ex: col_a + col_b)")
        if st.button("Aplicar transformações"):
            payload: Dict[str, Any] = {}
            if drop_columns:
                payload["drop_columns"] = drop_columns
            if fill_column:
                payload["fillna"] = {fill_column: fill_value}
            if add_name and add_expr:
                payload["add_columns"] = [{"name": add_name, "expression": add_expr}]
            response = api_post(f"/datasets/{dataset_id}/transform", payload)
            st.success("Transformações aplicadas")
            st.json(response)
    else:
        st.info("Faça upload de um dataset primeiro.")

if section == "Treinar":
    st.markdown("<div class='section-title'>4. Treinar modelos</div>", unsafe_allow_html=True)
    datasets = api_get("/datasets")
    dataset_map = {d["filename"] + f" ({d['dataset_id'][:8]})": d for d in datasets}
    if dataset_map:
        selection = st.selectbox("Dataset", list(dataset_map.keys()))
        dataset_id = dataset_map[selection]["dataset_id"]
        preview = api_get(f"/datasets/{dataset_id}/preview")
        df_preview = pd.DataFrame(preview)

        col_train, col_settings = st.columns([1.2, 0.8])
        with col_train:
            st.markdown(
                """
                <div class="card">
                    <h3>Configuração rápida</h3>
                    <p class="hint">Selecione o alvo e o tipo de problema. Nossa camada AutoML ajusta o pipeline ideal.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_settings:
            target = st.selectbox("Coluna target", df_preview.columns.tolist())
            problem_type = st.selectbox("Tipo de problema", ["classification", "regression", "time_series"])
            time_column = None
            if problem_type == "time_series":
                time_column = st.selectbox("Coluna de tempo (opcional)", [""] + df_preview.columns.tolist())
            if st.button("Treinar"):
                payload = {
                    "dataset_id": dataset_id,
                    "target": target,
                    "problem_type": problem_type,
                    "time_column": time_column or None,
                }
                response = api_post("/train", payload)
                st.success("Treino finalizado")
                st.json(response)
    else:
        st.info("Faça upload de um dataset primeiro.")

if section == "Modelos":
    st.markdown("<div class='section-title'>5. Modelos treinados e deploy</div>", unsafe_allow_html=True)
    models_payload = api_get("/models")
    models = list(models_payload.get("models", {}).values())
    if models:
        for model in models:
            st.markdown(
                """
                <div class="card">
                    <h3>Modelo treinado</h3>
                    <p class="hint">Detalhes completos e prontos para deploy.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.subheader(f"{model['algorithm']} ({model['model_id'][:8]})")
            st.json(model)

        st.markdown("<div class='section-title'>Deploy de modelo</div>", unsafe_allow_html=True)
        model_map = {f"{m['algorithm']} ({m['model_id'][:8]})": m for m in models}
        deploy_model_choice = st.selectbox("Modelo para deploy", list(model_map.keys()))
        deploy_name = st.text_input("Nome do deploy (ex: meu_modelo)", max_chars=40)
        if st.button("Publicar deploy"):
            payload = {
                "model_id": model_map[deploy_model_choice]["model_id"],
                "name": deploy_name,
            }
            try:
                result = api_post("/deployments", payload)
            except requests.HTTPError as exc:
                st.error(str(exc))
                st.stop()
            st.success("Deploy criado!")
            st.json(result)
            st.markdown(
                f"Abra o modelo em: `http://localhost:8501/{result['name']}`",
            )
    else:
        st.info("Nenhum modelo treinado ainda.")

    st.markdown("<div class='section-title'>Deploys publicados</div>", unsafe_allow_html=True)
    deployments_payload = api_get("/deployments")
    deployments = list(deployments_payload.get("deployments", {}).values())
    if deployments:
        deployment_map = {d["name"]: d for d in deployments}
        selected_deploy = st.selectbox("Selecione o deploy", list(deployment_map.keys()))
        st.json(deployment_map[selected_deploy])
        st.info("Acesse a aba 'Predições' para enviar arquivos e baixar previsões.")
    else:
        st.info("Nenhum deploy publicado ainda.")

if section == "Predições":
    st.markdown("<div class='section-title'>6. Predições em modelos deployados</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <h3>Central de predições</h3>
            <p class="hint">Escolha um deploy publicado, envie o CSV/Excel e baixe o arquivo final com previsões.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    deployments_payload = api_get("/deployments")
    deployments = list(deployments_payload.get("deployments", {}).values())
    if not deployments:
        st.info("Nenhum deploy publicado ainda.")
    else:
        deployment_map = {d["name"]: d for d in deployments}
        selected_deploy = st.selectbox("Deploy ativo", list(deployment_map.keys()))
        deployment = deployment_map[selected_deploy]

        col_left, col_right = st.columns([1.1, 0.9])
        with col_left:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{deployment['name']}</h3>
                    <p class="hint">Modelo: {deployment['algorithm']}</p>
                    <p class="hint">Target: {deployment.get('target') or 'N/A'}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            upload = st.file_uploader(
                "Envie o arquivo para predição",
                type=["csv", "xlsx", "xls"],
                key="deploy_upload_tab",
            )
            st.caption("A coluna target, se presente, será ignorada automaticamente.")
        with col_right:
            st.markdown(
                """
                <div class="card">
                    <h3>Como funciona</h3>
                    <ol>
                        <li>Faça upload do seu arquivo.</li>
                        <li>Executamos o modelo em segundos.</li>
                        <li>Baixe o CSV final com a coluna prediction.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if upload is not None and st.button("Executar predição", key="predict_tab"):
            try:
                result = api_post_file(
                    f"/deployments/{selected_deploy}/predict-file",
                    upload.name,
                    upload.getvalue(),
                )
            except requests.HTTPError as exc:
                st.error(str(exc))
                st.stop()
            st.success("Predição concluída")
            st.markdown(
                f"""
                <div class="prediction-result">
                    <h3>Resultado pronto</h3>
                    <div class="prediction-meta">
                        <span>Linhas previstas: <strong>{result['rows']}</strong></span>
                        <span>Arquivo: <strong>{result['output_filename']}</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            output_name = result.get("output_filename") or Path(result["output_file"]).name
            download_path = result.get("download_url") or f"/files/{output_name}"
            file_bytes = api_get_file(download_path)
            st.download_button(
                label="Baixar predição",
                data=file_bytes,
                file_name=output_name,
                mime="text/csv",
            )
