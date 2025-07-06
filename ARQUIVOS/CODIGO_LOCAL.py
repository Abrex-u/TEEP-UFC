import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scheptk.scheptk import FlowShop
import time
import math
import random
import sys
import os
import pickle
import pandas as pd
from rich.console import Console
from rich.table import Table

# =============================================================================
# FUNÇÕES DE APOIO E HEURÍSTICAS (Versão aprimorada)
# =============================================================================
def calcular_metricas_sequencia(sequencia, p_matrix, num_maquinas):
    num_trabalhos = len(sequencia)
    if num_trabalhos == 0: return 0, 0, None
    C = np.zeros((num_trabalhos, num_maquinas))
    C[0, 0] = p_matrix[sequencia[0], 0]
    for m in range(1, num_maquinas): C[0, m] = C[0, m-1] + p_matrix[sequencia[0], m]
    for j_seq in range(1, num_trabalhos):
        C[j_seq, 0] = C[j_seq-1, 0] + p_matrix[sequencia[j_seq], 0]
        for m in range(1, num_maquinas):
            C[j_seq, m] = max(C[j_seq-1, m], C[j_seq, m-1]) + p_matrix[sequencia[j_seq], m]
    makespan = C[-1, -1]
    return makespan, None, C

def calcular_makespan_sequencia(sequencia, p_matrix, num_maquinas):
    return calcular_metricas_sequencia(sequencia, p_matrix, num_maquinas)[0]

def calcular_limite_inferior(p_matrix, num_trabalhos, num_maquinas):
    soma_por_trabalho = np.sum(p_matrix, axis=1)
    lb_maquinas = 0
    for m in range(num_maquinas):
        cabecas = np.sum(p_matrix[:, :m], axis=1)
        caudas = np.sum(p_matrix[:, m+1:], axis=1)
        min_cabeca = np.min(cabecas) if m > 0 else 0
        min_cauda = np.min(caudas) if m < num_maquinas - 1 else 0
        soma_maquina_m = np.sum(p_matrix[:, m])
        lb_maquinas = max(lb_maquinas, min_cabeca + soma_maquina_m + min_cauda)
    return max(np.max(soma_por_trabalho), lb_maquinas)

def heuristica_frb5(p_matrix, num_trabalhos, num_maquinas):
    soma_tempos = np.sum(p_matrix, axis=1)
    trabalhos_ordenados = sorted(range(num_trabalhos), key=lambda k: soma_tempos[k], reverse=True)
    seq_parcial = [trabalhos_ordenados[0]]
    for i in range(1, num_trabalhos):
        job_a_inserir_k = trabalhos_ordenados[i]
        job_lookahead_k1 = trabalhos_ordenados[i+1] if i + 1 < num_trabalhos else None
        melhor_metrica, melhor_posicao = float('inf'), -1
        for pos_k in range(i + 1):
            seq_temp_k = seq_parcial[:pos_k] + [job_a_inserir_k] + seq_parcial[pos_k:]
            if job_lookahead_k1 is None:
                makespan_k, _, _ = calcular_metricas_sequencia(seq_temp_k, p_matrix, num_maquinas)
                metrica_atual = makespan_k
            else:
                min_makespan_lookahead = float('inf')
                for pos_k1 in range(i + 2):
                    seq_temp_k1 = seq_temp_k[:pos_k1] + [job_lookahead_k1] + seq_temp_k[pos_k1:]
                    makespan_k1 = calcular_makespan_sequencia(seq_temp_k1, p_matrix, num_maquinas)
                    if makespan_k1 < min_makespan_lookahead: min_makespan_lookahead = makespan_k1
                metrica_atual = min_makespan_lookahead
            if metrica_atual < melhor_metrica:
                melhor_metrica, melhor_posicao = metrica_atual, pos_k
        seq_parcial.insert(melhor_posicao, job_a_inserir_k)
    return seq_parcial
def heuristica_palmer(p_matrix, num_trabalhos, num_maquinas):
    indices = [(sum((num_maquinas - (2*i - 1)) * p_matrix[j, i-1] for i in range(1, num_maquinas + 1)), j) for j in range(num_trabalhos)]
    indices.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in indices]
def johnson_2_maquinas(p1, p2):
    n = len(p1); jobs = list(range(n))
    set1 = sorted([j for j in jobs if p1[j] < p2[j]], key=lambda j: p1[j])
    set2 = sorted([j for j in jobs if p1[j] >= p2[j]], key=lambda j: p2[j], reverse=True)
    return set1 + set2
def heuristica_cds(p_matrix, num_trabalhos, num_maquinas):
    melhor_sequencia, melhor_makespan = [], float('inf')
    for k in range(1, num_maquinas):
        p1, p2 = np.sum(p_matrix[:, :k], axis=1), np.sum(p_matrix[:, -k:], axis=1)
        sequencia_atual = johnson_2_maquinas(p1.tolist(), p2.tolist())
        makespan_atual = calcular_makespan_sequencia(sequencia_atual, p_matrix, num_maquinas)
        if makespan_atual < melhor_makespan: melhor_makespan, melhor_sequencia = makespan_atual, sequencia_atual
    return melhor_sequencia
def heuristica_gupta(p_matrix, num_trabalhos, num_maquinas):
    indices = []
    for j in range(num_trabalhos):
        e = 1 if p_matrix[j, 0] < p_matrix[j, num_maquinas - 1] else -1
        min_soma_adjacente = min(p_matrix[j, k] + p_matrix[j, k+1] for k in range(num_maquinas - 1))
        if min_soma_adjacente == 0: s_j = float('inf') 
        else: s_j = e / min_soma_adjacente
        indices.append((s_j, j))
    indices.sort(key=lambda x: x[0])
    return [idx for _, idx in indices]
def heuristica_neh_acelerada(p_matrix, num_trabalhos, num_maquinas):
    soma_tempos = np.sum(p_matrix, axis=1)
    trabalhos_ordenados = sorted(range(num_trabalhos), key=lambda k: soma_tempos[k], reverse=True)
    seq_parcial = [trabalhos_ordenados[0]]
    for i in range(1, num_trabalhos):
        job_a_inserir = trabalhos_ordenados[i]
        melhor_makespan, melhor_posicao = float('inf'), -1
        for k in range(i + 1):
            nova_seq = seq_parcial[:k] + [job_a_inserir] + seq_parcial[k:]
            makespan = calcular_makespan_sequencia(nova_seq, p_matrix, num_maquinas)
            if makespan < melhor_makespan: melhor_makespan, melhor_posicao = makespan, k
        seq_parcial.insert(melhor_posicao, job_a_inserir)
    return seq_parcial

# =============================================================================
# BUSCA LOCAL (Intensificação com "Best Improvement")
# =============================================================================
def avaliar_makespan_parcial(sequencia, p_matrix, num_maquinas, start_job_idx, C_anterior):
    C = np.copy(C_anterior); n_seq = len(sequencia)
    for j_seq in range(start_job_idx, n_seq):
        job_idx = sequencia[j_seq]
        c_acima_m0 = C[j_seq - 1, 0] if j_seq > 0 else 0
        C[j_seq, 0] = c_acima_m0 + p_matrix[job_idx, 0]
        for m in range(1, num_maquinas):
            c_acima = C[j_seq - 1, m] if j_seq > 0 else 0
            C[j_seq, m] = max(c_acima, C[j_seq, m - 1]) + p_matrix[job_idx, m]
    return C[-1, -1], C
def calcular_matriz_c_parcial(seq, p_matrix, num_maquinas):
    c = np.zeros((len(seq), num_maquinas))
    if not seq: return c
    c[0, 0] = p_matrix[seq[0], 0]
    for m in range(1, num_maquinas): c[0, m] = c[0, m-1] + p_matrix[seq[0], m]
    for i in range(1, len(seq)):
        c[i, 0] = c[i-1, 0] + p_matrix[seq[i], 0]
        for m in range(1, num_maquinas):
            c[i, m] = max(c[i-1, m], c[i, m-1]) + p_matrix[seq[i], m]
    return c
def calcular_matriz_q_parcial(seq, p_matrix, num_maquinas):
    q = np.zeros((len(seq), num_maquinas)); n_seq = len(seq)
    if not seq: return q
    q[n_seq-1, num_maquinas-1] = p_matrix[seq[n_seq-1], num_maquinas-1]
    for m in range(num_maquinas-2, -1, -1): q[n_seq-1, m] = q[n_seq-1, m+1] + p_matrix[seq[n_seq-1], m]
    for i in range(n_seq-2, -1, -1):
        q[i, num_maquinas-1] = q[i+1, num_maquinas-1] + p_matrix[seq[i], num_maquinas-1]
        for m in range(num_maquinas-2, -1, -1):
            q[i, m] = max(q[i+1, m], q[i, m+1]) + p_matrix[seq[i], m]
    return q
def explorar_vizinhanca_swap_best_improvement(seq, p_matrix, num_maquinas, atual_best_m, C_matrix):
    melhor_vizinho_seq, melhor_vizinho_m = seq, atual_best_m
    melhor_vizinho_C = C_matrix
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            nova_seq = seq[:]; nova_seq[i], nova_seq[j] = nova_seq[j], nova_seq[i]
            novo_makespan, nova_C = avaliar_makespan_parcial(nova_seq, p_matrix, num_maquinas, i, C_matrix)
            if novo_makespan < melhor_vizinho_m:
                melhor_vizinho_m, melhor_vizinho_seq, melhor_vizinho_C = novo_makespan, nova_seq[:], nova_C
    houve_melhora = melhor_vizinho_m < atual_best_m
    return melhor_vizinho_seq, melhor_vizinho_m, melhor_vizinho_C, houve_melhora
def explorar_vizinhanca_insertion_best_improvement(seq, p_matrix, num_maquinas, atual_best_m, C_matrix):
    melhor_melhora = atual_best_m; melhor_nova_seq = seq
    for i in range(len(seq)):
        job_a_mover = seq[i]; seq_parcial = seq[:i] + seq[i+1:]
        c_parcial = calcular_matriz_c_parcial(seq_parcial, p_matrix, num_maquinas); q_parcial = calcular_matriz_q_parcial(seq_parcial, p_matrix, num_maquinas)
        for k in range(len(seq)):
            c_job = np.zeros(num_maquinas); prev_c_job = c_parcial[k-1] if k > 0 else np.zeros(num_maquinas)
            c_job[0] = prev_c_job[0] + p_matrix[job_a_mover, 0]
            for m in range(1, num_maquinas): c_job[m] = max(c_job[m-1], prev_c_job[m]) + p_matrix[job_a_mover, m]
            next_q_job = q_parcial[k] if k < len(seq_parcial) else np.zeros(num_maquinas)
            makespan_teste = np.max(c_job + next_q_job)
            if makespan_teste < melhor_melhora:
                 melhor_melhora = makespan_teste; melhor_nova_seq = seq_parcial[:k] + [job_a_mover] + seq_parcial[k:]
    if melhor_melhora < atual_best_m:
        novo_makespan, _, nova_C = calcular_metricas_sequencia(melhor_nova_seq, p_matrix, num_maquinas)
        return melhor_nova_seq, novo_makespan, nova_C, True
    return seq, atual_best_m, C_matrix, False
def busca_local_vns(sequencia, p_matrix, num_maquinas):
    melhor_sequencia = sequencia[:]; melhor_makespan, _, C_matrix = calcular_metricas_sequencia(melhor_sequencia, p_matrix, num_maquinas)
    vizinhancas = [explorar_vizinhanca_insertion_best_improvement, explorar_vizinhanca_swap_best_improvement]
    i_vizinhanca = 0
    while i_vizinhanca < len(vizinhancas):
        nova_sequencia, novo_makespan, nova_C_matrix, houve_melhora = vizinhancas[i_vizinhanca](melhor_sequencia, p_matrix, num_maquinas, melhor_makespan, C_matrix)
        if houve_melhora:
            melhor_sequencia, melhor_makespan, C_matrix = nova_sequencia, novo_makespan, nova_C_matrix
            i_vizinhanca = 0
        else:
            i_vizinhanca += 1
    return melhor_sequencia, melhor_makespan

# =============================================================================
# <<< PROJETO ILS: NOVO MOTOR DE OTIMIZAÇÃO >>>
# =============================================================================
def perturbacao(sequencia, forca):
    """
    Aplica uma perturbação na sequência para escapar de ótimos locais.
    A 'forca' determina quantos movimentos de troca (swap) aleatórios são feitos.
    """
    seq_perturbada = sequencia[:]
    num_trabalhos = len(seq_perturbada)
    for _ in range(forca):
        i, j = random.sample(range(num_trabalhos), 2)
        seq_perturbada[i], seq_perturbada[j] = seq_perturbada[j], seq_perturbada[i]
    return seq_perturbada

def motor_ils(solucao_inicial, makespan_inicial, p_matrix, num_maquinas, tempo_limite):
    """
    Motor principal baseado na arquitetura de Busca Local Iterada (ILS).
    """
    start_time = time.time()
    iteracao = 0
    
    melhor_seq_global = solucao_inicial[:]
    makespan_global = makespan_inicial
    
    # Parâmetros do ILS Adaptativo
    iter_sem_melhora = 0
    forca_perturbacao_base = 4  # Começa com uma perturbação leve
    forca_perturbacao_atual = forca_perturbacao_base

    print(f"   Iniciando motor ILS. Melhor makespan inicial: {makespan_global:.2f}")

    try:
        while (time.time() - start_time) < tempo_limite:
            iteracao += 1
            
            # 1. Perturbação: Aplica um "chute" na melhor solução encontrada até agora
            seq_perturbada = perturbacao(melhor_seq_global, forca=forca_perturbacao_atual)
            
            # 2. Busca Local: Aplica a VNS intensiva na solução perturbada
            nova_seq, novo_makespan = busca_local_vns(seq_perturbada, p_matrix, num_maquinas)
            
            # 3. Critério de Aceitação
            if novo_makespan < makespan_global:
                melhor_seq_global, makespan_global = nova_seq[:], novo_makespan
                iter_sem_melhora = 0
                forca_perturbacao_atual = forca_perturbacao_base # Reset da força após melhora
                print(f"\r   [Iter: {iteracao}] 🎉 NOVO MELHOR GLOBAL: {makespan_global:.2f} (Força Pert.={forca_perturbacao_atual})", end="")
            else:
                iter_sem_melhora += 1
            
            # Lógica Adaptativa: Aumenta a força da perturbação se a busca estagnar
            if iter_sem_melhora > 150:
                forca_perturbacao_atual += 2
                iter_sem_melhora = 0 # Reseta o contador para dar tempo para a nova força funcionar
                print(f"\n   [Iter: {iteracao}] Estagnação detectada. Aumentando força da perturbação para {forca_perturbacao_atual}...")

            if iteracao % 25 == 0:
                print(f"\r   [Iter: {iteracao}] Buscando... Melhor: {makespan_global:.2f} (Força Pert.={forca_perturbacao_atual})", end="")

    except KeyboardInterrupt:
        print("\n\n🛑 Busca interrompida pelo usuário.")

    print(f"\n   Busca ILS finalizada. Tempo decorrido: {time.time() - start_time:.2f}s, Iterações: {iteracao}")
    return melhor_seq_global, makespan_global

# =============================================================================
# ORQUESTRADOR DE EXPERIMENTOS (Adaptado para o ILS)
# =============================================================================
def gerar_grafico_gantt(sequencia, p_matrix, num_maquinas, makespan, nome_base_instancia=None, caminho_salvar=None):
    num_trabalhos = len(sequencia)
    C = np.zeros((num_trabalhos, num_maquinas))
    S = np.zeros((num_trabalhos, num_maquinas))
    for j_seq, job_idx in enumerate(sequencia):
        for m in range(num_maquinas):
            S[j_seq, m] = max(C[j_seq, m-1] if m > 0 else 0, C[j_seq-1, m] if j_seq > 0 else 0)
            C[j_seq, m] = S[j_seq, m] + p_matrix[job_idx, m]

    fig, ax = plt.subplots(figsize=(18, 7))
    cores = plt.cm.tab20(np.linspace(0, 1, num_trabalhos))
    yticks = []
    yticklabels = []
    altura = 0.8

    for m in range(num_maquinas):
        for j_seq, job_idx in enumerate(sequencia):
            start = S[j_seq, m]
            dur = p_matrix[job_idx, m]
            rect = patches.Rectangle(
                (start, m - altura/2), dur, altura,
                facecolor=cores[job_idx % len(cores)],
                edgecolor='black', linewidth=1.2, alpha=0.95
            )
            ax.add_patch(rect)
            # Label do job
            if dur > 0.04 * makespan or num_trabalhos <= 30:
                ax.text(
                    start + dur/2, m,
                    f'T{job_idx+1}',
                    ha='center', va='center',
                    color='white' if dur > 0.08 * makespan else 'black',
                    fontsize=10, fontweight='bold'
                )
        yticks.append(m)
        yticklabels.append(f'Máquina {m+1}')

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=12)
    ax.set_xlabel('Tempo', fontsize=13, weight='bold')
    ax.set_ylabel('Recursos', fontsize=13, weight='bold')
    titulo = f'Gráfico de Gantt da Melhor Solução\nMakespan = {makespan:.0f}'
    if nome_base_instancia is not None:
        titulo += f' | {nome_base_instancia}'
    ax.set_title(titulo, fontsize=18, weight='bold', pad=20)
    ax.grid(axis='x', linestyle=':', color='gray', alpha=0.7)
    ax.set_xlim(0, makespan * 1.03)
    ax.set_ylim(-0.7, num_maquinas - 0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    # Legenda
    legend_handles = [
        patches.Patch(color=cores[job_idx % len(cores)], label=f'T{job_idx+1}')
        for job_idx in sequencia[:min(15, num_trabalhos)]
    ]
    if num_trabalhos > 15:
        legend_handles.append(patches.Patch(color='none', label='...'))
    ax.legend(handles=legend_handles, title='Trabalhos', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, title_fontsize=12, frameon=True)
    plt.tight_layout()
    if caminho_salvar is not None:
        plt.savefig(caminho_salvar, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def executar_solver_para_instancia(caminho_instancia, tempo_limite, modo_turbo, mostrar_gantt=False, salvar_gantt_path=None):
    tempo_execucao_total_inicio = time.time()
    nome_base_instancia = os.path.basename(caminho_instancia)
    try:
        instance = FlowShop(caminho_instancia); num_trabalhos, num_maquinas = instance.jobs, instance.machines
        p_matrix = np.array(instance.pt).T
        print(f"   Instância carregada: {nome_base_instancia} ({num_trabalhos} jobs, {num_maquinas} máquinas)")
    except Exception as e:
        print(f"❌ ERRO CRÍTICO ao carregar a instância {caminho_instancia}: {e}"); return None

    print("   Iniciando Estágio 1: Geração da solução inicial de elite...")
    sementes = {'Palmer': heuristica_palmer(p_matrix, num_trabalhos, num_maquinas),'CDS': heuristica_cds(p_matrix, num_trabalhos, num_maquinas),'Gupta': heuristica_gupta(p_matrix, num_trabalhos, num_maquinas),'NEH': heuristica_neh_acelerada(p_matrix, num_trabalhos, num_maquinas),'FRB5': heuristica_frb5(p_matrix, num_trabalhos, num_maquinas)}
    campeao_inicial_seq, campeao_inicial_mks = (None, float('inf'))
    for nome, seq_inicial in sementes.items():
        # Aplica uma busca local completa na semente
        seq_refinada, mks_refinado = busca_local_vns(seq_inicial, p_matrix, num_maquinas)
        if mks_refinado < campeao_inicial_mks:
            campeao_inicial_seq, campeao_inicial_mks = seq_refinada, mks_refinado
    print(f"   🏆 Ponto de Partida de Elite Encontrado: Makespan = {campeao_inicial_mks:.2f}")
    
    # <<< PROJETO ILS: Chamando o novo motor >>>
    tempo_restante = tempo_limite - (time.time() - tempo_execucao_total_inicio)
    if tempo_restante <= 0:
        print("   Tempo esgotado durante a fase inicial.")
        seq_final, makespan_final = campeao_inicial_seq, campeao_inicial_mks
    else:
        seq_final, makespan_final = motor_ils(
            campeao_inicial_seq, campeao_inicial_mks, p_matrix, num_maquinas, tempo_restante
        )

    limite_inferior = calcular_limite_inferior(p_matrix, num_trabalhos, num_maquinas)
    gap = ((makespan_final - limite_inferior) / limite_inferior) * 100 if limite_inferior > 0 else 0
    tempo_total_execucao = time.time() - tempo_execucao_total_inicio
    resultado = {"Instância": nome_base_instancia, "Makespan Inicial": campeao_inicial_mks, "Makespan Final": makespan_final, "Limite Inferior (LB)": limite_inferior, "GAP (%)": gap, "Tempo Total (s)": tempo_total_execucao, "Sequência Final": seq_final}
    if mostrar_gantt:
        gerar_grafico_gantt(seq_final, p_matrix, num_maquinas, makespan_final, nome_base_instancia)
    elif salvar_gantt_path:
        caminho_completo_salvar = os.path.join(salvar_gantt_path, f"gantt_{nome_base_instancia.replace('.txt', '')}.png")
        gerar_grafico_gantt(seq_final, p_matrix, num_maquinas, makespan_final, nome_base_instancia, caminho_salvar=caminho_completo_salvar)
    return resultado

def modo_batch(caminho_pasta, lista_arquivos):
    print("\n--- MODO BATCH ---")
    while True:
        try:
            tempo = int(input("Digite o tempo limite EM SEGUNDOS para CADA instância: "));
            if tempo > 0: break
            else: print("O tempo deve ser um número positivo.")
        except ValueError: print("Entrada inválida. Digite um número inteiro.")
    pasta_gantts = f"gantts_resultados_{int(time.time())}"; os.makedirs(pasta_gantts, exist_ok=True)
    print(f"\nOs gráficos de Gantt serão salvos na pasta: '{pasta_gantts}/'")
    resultados_finais = []
    for nome_arquivo in lista_arquivos:
        caminho_completo = os.path.join(caminho_pasta, nome_arquivo)
        print(f"\n================ Processando: {nome_arquivo} ================")
        resultado = executar_solver_para_instancia(caminho_completo, tempo_limite=tempo, modo_turbo=False, salvar_gantt_path=pasta_gantts)
        if resultado:
            resultados_finais.append(resultado)
            print(f"   Resultado: Makespan Final = {resultado['Makespan Final']:.2f} | GAP = {resultado['GAP (%)']:.2f}%")
        print("=" * (len(nome_arquivo) + 28))
    if resultados_finais:
        console = Console(); table = Table(title="🔬 Resumo Completo dos Resultados - PROJETO ILS 🔬", show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Instância", justify="left", style="cyan", no_wrap=True); table.add_column("M. Inicial", justify="right", style="magenta"); table.add_column("M. Final", justify="right", style="bold green"); table.add_column("LB", justify="right", style="magenta"); table.add_column("GAP (%)", justify="right"); table.add_column("Tempo (s)", justify="right", style="yellow"); table.add_column("Sequência Final", justify="left", style="dim")
        for res in resultados_finais:
            gap = res["GAP (%)"]
            if gap < 1.0: gap_style = "bold green"
            elif gap < 3.0: gap_style = "yellow"
            else: gap_style = "bold red"
            table.add_row(res["Instância"], f"{res['Makespan Inicial']:.0f}", f"{res['Makespan Final']:.0f}", f"{res['Limite Inferior (LB)']:.0f}", f"[{gap_style}]{gap:.2f}%[/{gap_style}]", f"{res['Tempo Total (s)']:.2f}", '-'.join(map(str, [s + 1 for s in res['Sequência Final']])))
        print("\n\n"); console.print(table)
        df = pd.DataFrame(resultados_finais); df['Sequência Final'] = df['Sequência Final'].apply(lambda seq: '-'.join(map(str, [s + 1 for s in seq])))
        nome_csv = f"resultados_batch_completos_{int(time.time())}.csv"; df.to_csv(nome_csv, index=False); print(f"\n[bold green]Resultados completos salvos em arquivo:[/] '{nome_csv}'")

def modo_isolado(caminho_pasta, lista_arquivos):
    print("\n--- MODO ISOLADO ---"); print("Instâncias disponíveis:")
    for i, nome in enumerate(lista_arquivos): print(f"  {i+1}: {nome}")
    while True:
        try:
            escolha = int(input(f"Escolha o número da instância (1-{len(lista_arquivos)}): "));
            if 1 <= escolha <= len(lista_arquivos): break
            else: print("Escolha fora do intervalo.")
        except ValueError: print("Entrada inválida.")
    caminho_instancia_escolhida = os.path.join(caminho_pasta, lista_arquivos[escolha-1])
    resp_tempo = input("Digite o tempo limite em segundos (ou deixe em branco para busca incansável): ")
    tempo = int(resp_tempo) if resp_tempo.isdigit() and int(resp_tempo) > 0 else 999999
    print(f"\n================ Processando: {os.path.basename(caminho_instancia_escolhida)} ================")
    resultado = executar_solver_para_instancia(caminho_instancia_escolhida, tempo_limite=tempo, modo_turbo=False, mostrar_gantt=True)
    if resultado:
        print("\n--- 👑 RESULTADO FINAL ---"); print(f"Instância: {resultado['Instância']}"); print(f"Makespan Final: {resultado['Makespan Final']:.2f}"); print(f"GAP (%): {resultado['GAP (%)']:.2f}%"); print(f"Tempo Total (s): {resultado['Tempo Total (s)']:.2f}")
        seq_formatada = '-'.join(map(str, [s + 1 for s in resultado['Sequência Final']])); print(f"Sequência Final: {seq_formatada}")

if __name__ == '__main__':
    CAMINHO_INSTANCIAS = r'CAMINHO DA PASTA COM AS INSTÂNCIAS'  
    try:
        arquivos_instancia = sorted([f for f in os.listdir(CAMINHO_INSTANCIAS) if f.endswith('.txt')])
        if not arquivos_instancia: print(f"❌ Nenhum arquivo .txt encontrado em '{CAMINHO_INSTANCIAS}'"); sys.exit()
    except FileNotFoundError: print(f"❌ ERRO: O diretório não foi encontrado: '{CAMINHO_INSTANCIAS}'"); sys.exit()
    print("--- 🔬 PAINEL DE CONTROLE - PROJETO ILS 🔬 ---"); print(f"{len(arquivos_instancia)} instâncias encontradas.")
    print("\nEscolha o modo de execução:"); print("  1: Rodar todas as instâncias (Modo Batch)"); print("  2: Escolher uma instância específica (Modo Isolado)")
    while True:
        opcao = input("Digite sua opção (1 ou 2): ")
        if opcao == '1': modo_batch(CAMINHO_INSTANCIAS, arquivos_instancia); break
        elif opcao == '2': modo_isolado(CAMINHO_INSTANCIAS, arquivos_instancia); break
        else: print("Opção inválida. Tente novamente.")