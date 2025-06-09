import os
from queue import Queue
import signal
import threading
import multiprocessing
import time
from datetime import datetime, timedelta
from typing import Callable, Any, Dict, List
from biblioteca.modulos import banco

import psutil

class Tasker:

    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implementa o padrão Singleton garantindo que apenas uma instância exista."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Tasker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Inicializa o gerenciador de tarefas."""
        if not hasattr(self, 'initialized'):  # Para evitar reinicialização do singleton
            self.processos: Dict[str, Dict[str, Any]] = {}  # Dicionário para controle de processos por usuário
            self.queue = Queue()  # Fila para armazenar as tarefas
            self.initialized = True

    def nova_task(self, funcao: Callable[..., Any], cron: str = None, reduzir_carga_cpu: bool = True, usuario_id: str = '', *args, **kwargs):
        return
        """
        Cria uma nova tarefa a ser gerenciada pelo Tasker.

        :param funcao: Função a ser executada em background.
        :param cron: Cronograma no formato cron para agendamento.
        :param reduzir_carga_cpu: Se True, reduz a prioridade do processo para diminuir o uso de CPU.
        :param usuario_id: ID do usuário para controle e identificação dos processos.
        :param args: Argumentos da função.
        :param kwargs: Argumentos nomeados da função.
        """
        # Encerra o processo ativo do usuário, se existir

        if cron:
            self.agendar_tarefa(cron, funcao, usuario_id)
        else:
            self.adicionar_tarefa(funcao, usuario_id, *args, **kwargs)

        if reduzir_carga_cpu:
            self.reduzir_prioridade_processo()

        if not hasattr(self, 'thread'):
            self.thread = threading.Thread(target=self.run)
            self.thread.daemon = True  # Define a thread como um daemon para que não impeça a finalização do programa principal
            self.thread.start()

    def adicionar_tarefa(self, funcao: Callable[..., Any], usuario_id: str, *args, **kwargs):
        """Adiciona uma tarefa à fila."""
        self.queue.put((funcao, usuario_id, args, kwargs))
        print(f"Tarefa {funcao.__name__} adicionada à fila{f' para o usuário {usuario_id}' if usuario_id else ''}.")

    def processar_tarefa(self):
        """Processa uma tarefa da fila."""
        while not self.queue.empty():
            funcao, usuario_id, args, kwargs = self.queue.get()
            self.encerrar_processos(usuario_id, False)
            # Cria um novo processo para a tarefa usando _processando_tarefa
            process = multiprocessing.Process(target=self._processando_tarefa, args=(funcao, usuario_id) + args, kwargs=kwargs)
            process.start()
            # Obter nome da função
            self._adicionar_processo(process, usuario_id, funcao.__name__)
                
    def _processando_tarefa(self, funcao: Callable[..., Any], usuario_id: str, *args, **kwargs):
        """Processa uma tarefa da fila."""
        try:
            funcao(*args, **kwargs)
            print(f"Encerrando tarefa {funcao.__name__} com sucesso.")
            self.encerrar_processos(usuario_id, True)
        except Exception as e:
            print(f"Erro ao processar a tarefa {funcao.__name__}: {e}")

    def _adicionar_processo(self, process: multiprocessing.Process, usuario_id: str, funcao: str):
        # """Adiciona um processo ao dicionário de processos."""
        # if usuario_id not in self.processos:
        #     self.processos[usuario_id] = {'processes': []}
        # self.processos[usuario_id]['processes'].append(process)
        print(f"<{process.pid}> {funcao} iniciado{f' para o usuário {usuario_id}' if usuario_id else ''}.")
        # adicionar a tabela processo o nome da funcao, usuario_id e o pid
        banco().sql("INSERT INTO processo (nome, pid_processo, id_usuario, data_inicio) VALUES ($1, $2, $3, NOW()) returning pid_processo", (funcao, process.pid, usuario_id))

    def _remover_processo(self, pid_processo, status_concluido: bool):
        print(f"Removendo processo <{pid_processo}>.")
        """Remove um processo do dicionário de processos e do banco() de dados."""
        # Obter detalhes do processo
        query_get = """
            SELECT nome, data_inicio
            FROM processo
            WHERE pid_processo = $1
        """
        process_details = banco().sql(query_get, (pid_processo,))
        
        if process_details:
            nome = process_details[0]['nome']
        
            data_inicio = process_details[0]['data_inicio']
        
            # Calcular o fuso horário de Brasília (UTC-3)
            utc_offset = timedelta(hours=-3)

            # Ajustar data_inicio para o fuso horário de Brasília
            if isinstance(data_inicio, datetime):
                data_inicio = data_inicio + utc_offset

            # Obter a hora atual ajustada para o fuso horário de Brasília
            data_fim = datetime.now()
            duracao = data_fim - data_inicio

            # Inserir no processos_completos
            query_insert = """
                INSERT INTO processos_completos (nome, data_inicio, data_fim, duracao, status)
                VALUES ($1, $2, $3, $4, $5) returning id
            """
            print(f"Processo <{pid_processo}> [{nome}] concluído em {duracao} com status: {status_concluido}.")
            banco().sql(query_insert, (nome, data_inicio, data_fim, duracao, status_concluido))
        
        pid_processo = banco().sql("DELETE FROM processo WHERE pid_processo = $1 returning pid_processo", (pid_processo))
        
        if not pid_processo:
            print(f"Processo <{pid_processo}> não encontrado no banco de dados.")
            return
        else:
            pid_processo = pid_processo[0]['pid_processo']
        print(f"Processo <{pid_processo}> removido do banco de dados.")

    def encerrar_processos(self, usuario_id: str = None, concluido: bool = False):
        """Encerra todos os processos ou os processos de um usuário específico."""
        if usuario_id == "TODOS":
            processos = banco().sql("SELECT pid_processo, id_usuario, nome FROM processo", ())
            print(f"Processos: {processos}")
            for processo in processos:
                self._encerrar_processo(processo['pid_processo'], processo['id_usuario'], processo['nome'], concluido)
            print("Todos os processos foram encerrados.")
        else:
            processos = banco().sql("SELECT pid_processo, id_usuario, nome FROM processo WHERE id_usuario = $1", (usuario_id,))
            print(f"Processos: {processos}")
            for processo in processos:
                self._encerrar_processo(processo['pid_processo'], processo['id_usuario'], processo['nome'], concluido)
            #print(f"Todos os processos para o usuário {usuario_id} encerrados.")
            
    def _encerrar_processo(self, pid_processo: int, usuario_id: str, nome_funcao: str, concluido: bool = False):
        """Encerra um processo específico pelo PID."""
        process = None
        for p in multiprocessing.active_children():
            if str(p.pid) == str(pid_processo):
                process = p
                break
        if not process:
            print(f"<{pid_processo}> [{nome_funcao}] Processo não encontrado.")
            self._remover_processo(pid_processo, concluido)
            return
        if process.is_alive():
            print(f"<{pid_processo}> [{nome_funcao}] Sendo encerrado para o usuário {usuario_id}.")
            process.terminate()
            process.join(timeout=2)  # Espera alguns segundos para ver se termina

            if process.is_alive():
                os.kill(pid_processo, signal.SIGKILL)  # Força a finalização se ainda estiver vivo
                print(f"<{pid_processo}> [{nome_funcao}] Finalizado forçadamente para o usuário {usuario_id}.")
            else:
                print(f"<{pid_processo}> [{nome_funcao}] Finalizado para o usuário {usuario_id}.")
            # Remover do banco() de dados
            self._remover_processo(pid_processo, concluido)

        

    def agendar_tarefa(self, cron: str, funcao: Callable[..., Any], usuario_id: str):
        """
        Agenda a execução da função de acordo com a expressão cron.

        :param cron: Expressão cron (por exemplo, "0 12 * * *" para executar todos os dias ao meio-dia).
        :param funcao: Função a ser agendada.
        :param usuario_id: ID do usuário que está agendando a tarefa.
        """
        if not self._validar_cron(cron):
            print("Expressão cron inválida.")
            return
        
        proxima_execucao = self._proximo_horario_execucao(cron)
        #print(f"Tarefa {funcao.__name__} agendada com a expressão cron: {cron}{f' para o usuário {usuario_id}' if usuario_id else ''}. Próxima execução: {proxima_execucao}")

        # Aqui criamos uma thread específica para lidar com o agendamento
        agendamento_thread = threading.Thread(target=self._agendamento_cron, args=(cron, funcao, usuario_id))
        agendamento_thread.daemon = True  # Define como daemon para que não impeça a finalização do programa principal
        agendamento_thread.start()

    def _agendamento_cron(self, cron: str, funcao: Callable[..., Any], usuario_id: str):
        """Gerencia o agendamento cron."""
        while True:
            proxima_execucao = self._proximo_horario_execucao(cron)
            now = datetime.now()
            sleep_time = (proxima_execucao - now).total_seconds()
            print(f"Aguardando {sleep_time} segundos até a próxima execução de {funcao.__name__}{f' para o usuário {usuario_id}' if usuario_id else ''}.")
            time.sleep(sleep_time)

            self.adicionar_tarefa(funcao, usuario_id)
            self.processar_tarefa()

    @staticmethod
    def _validar_cron(cron: str) -> bool:
        """Valida a expressão cron."""
        partes_cron = cron.split()
        if len(partes_cron) != 5:
            return False
        return all(partes_cron[i].isdigit() or partes_cron[i] == '*' for i in range(5))

    def _proximo_horario_execucao(self, cron: str) -> datetime:
        """Calcula o próximo horário de execução baseado na expressão cron."""
        now = datetime.now()
        minuto, hora, dia, mes, dia_semana = cron.split()

        ano = now.year
        if mes == '*':
            mes = now.month
        if dia == '*':
            dia = now.day
        if dia_semana != '*':
            dia_semana = (now.weekday() + int(dia_semana) - now.weekday()) % 7
            dia = now.day + (dia_semana - now.weekday()) % 7

        proxima_execucao = datetime(ano, int(mes), int(dia), int(hora), int(minuto))

        if proxima_execucao <= now:
            proxima_execucao = proxima_execucao + timedelta(days=1)
        
        return proxima_execucao

    @staticmethod
    def reduzir_prioridade_processo():
        """Reduz a prioridade do processo para diminuir a carga da CPU."""
        try:
            if os.name == 'nt':  # Windows
                p = psutil.Process(os.getpid())
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                p = psutil.Process(os.getpid())
                p.nice(10)
        except Exception as e:
            print(f"Erro ao reduzir a prioridade do processo: {e}")

    def run(self):
        """Executa o loop principal para processar tarefas."""
        while True:
            self.processar_tarefa()
            time.sleep(1)
