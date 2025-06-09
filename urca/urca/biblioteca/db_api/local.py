import pandas as pd
import os
from .abstract_db import AbstractDB
from .local_data import CurvaForward, DemandaMaxima, CMOSemanal,       \
                        ProjCapInstalada, EnergiaVertidaTurbinavel,    \
                        PojecaoCarga, GeracaoUsinasDespachadas,        \
                        IntercambioNacional, IntercambioInternacional, \
                        IndicadorPrevisaoEna, PisoTetoPLD, PLDMensal,  \
                        FeriadosNacionais, \
                        AmpereEnaHistRee, AmpereEnaAtualRee,           \
                        AmperePrecipitacaoHistoricaREE,                \
                        AmperePrecipitacaoHistoricaBacias,             \
                        AmperePrevisaoHistoricaREE, AmpereIndiceMensalAMO, \
                        AmperePrevisaoHistoricaBacias, AmpereClimatologiaEnaBacia,\
                        AmpereEnaHistBacia, AmpereEnaAtualBacia, \
                        AmpereClimatologiaPrecipitacaoBacia, \
                        AmpereClimatologiaPrecipitacaoREE, \
                        AmpereClimatologiaEnaREE, AmpereIndiceDiarioAAO, \
                        AmpereIndiceMensalAAO, AmpereIndiceMensalIOD, \
                        AmpereIndiceSemanalIOD, AmpereIndiceDiarioMJO, \
                        AmpereIndiceMensalONI, AmpereIndiceTrimestralONI, \
                        AmperePrevClimaticaREE, AmperePrevClimaticaBacia, \
                        AmpereEnaPrevREE, AmpereEnaPrevBacia, AmpereZCAS, \
                        AmpereASAS, AmpereJBN, AmpereFrentesFrias,        \
                        AmpereIndiceMensalPDO, AmpereIndiceMensalTNA, \
                        AmpereIndiceMensalTSA, AmpereIndiceMensalNINOS, \
                        AmperePrevClimaticaComplementoREE, \
                        AmperePrevClimaticaComplementoBacia, \
                        AmperePrevClimaticaComplementoCPCREE, \
                        AmperePrevClimaticaComplementoCPCBacia, \
                        AmperePrevClimaticaHistMensalENA, \
                        AmperePrevClimaticaItem3e9_4REESUniao, \
                        AmperePrevClimaticaItem3e9_4BaciaUniao, \
                        CMOSemihorario, AmpereIndiceMensalComplementoAMO, \
                        AmpereIndiceMensalComplementoPDO, \
                        AmpereIndiceMensalComplementoTNA, \
                        AmpereIndiceMensalComplementoTSA, \
                        AmpereIndiceMensalComplementoNINOS, \
                        AmpereIndiceMensalSAD, \
                        AmpereIndiceMensalPrevisao24TNA, \
                        AmpereIndiceMensalPrevisao24TSA, \
                        AmpereIndiceMensalPrevisao24NINOS, \
                        AmpereEnaHistMensal, AmpereClimatologiaEnaSUBs, \
                        AmpereEARdSubs, UpdatedAmpereEnaPrevREE, PojecaoCargaRaw, \
                        AmpereEnaHistReeNew


class Local(AbstractDB):
    """
        Esta classe faz a interface entre das classes em local_data
    """
    __TABLES = {
        CurvaForward.name: CurvaForward,
        DemandaMaxima.name: DemandaMaxima,
        CMOSemanal.name: CMOSemanal,
        ProjCapInstalada.name: ProjCapInstalada,
        EnergiaVertidaTurbinavel.name: EnergiaVertidaTurbinavel,
        PojecaoCarga.name: PojecaoCarga,
        GeracaoUsinasDespachadas.name: GeracaoUsinasDespachadas,
        IntercambioNacional.name: IntercambioNacional,
        IntercambioInternacional.name: IntercambioInternacional,
        IndicadorPrevisaoEna.name: IndicadorPrevisaoEna,
        PisoTetoPLD.name: PisoTetoPLD,
        FeriadosNacionais.name: FeriadosNacionais,
        PLDMensal.name: PLDMensal,
        AmpereEnaHistRee.name: AmpereEnaHistRee,
        AmpereEnaAtualRee.name: AmpereEnaAtualRee,
        AmperePrecipitacaoHistoricaREE.name: AmperePrecipitacaoHistoricaREE,
        AmperePrecipitacaoHistoricaBacias.name: AmperePrecipitacaoHistoricaBacias,
        AmperePrevisaoHistoricaREE.name: AmperePrevisaoHistoricaREE,
        AmperePrevisaoHistoricaBacias.name: AmperePrevisaoHistoricaBacias,
        AmpereEnaHistBacia.name: AmpereEnaHistBacia,
        AmpereEnaAtualBacia.name: AmpereEnaAtualBacia,
        AmpereClimatologiaEnaBacia.name: AmpereClimatologiaEnaBacia,
        AmpereClimatologiaPrecipitacaoBacia.name: AmpereClimatologiaPrecipitacaoBacia,
        AmpereClimatologiaPrecipitacaoREE.name: AmpereClimatologiaPrecipitacaoREE,
        AmpereClimatologiaEnaREE.name: AmpereClimatologiaEnaREE,
        AmpereIndiceDiarioAAO.name: AmpereIndiceDiarioAAO,
        AmpereIndiceMensalAAO.name: AmpereIndiceMensalAAO,
        AmpereIndiceMensalIOD.name: AmpereIndiceMensalIOD,
        AmpereIndiceSemanalIOD.name: AmpereIndiceSemanalIOD,
        AmpereIndiceDiarioMJO.name: AmpereIndiceDiarioMJO,
        AmpereIndiceMensalONI.name: AmpereIndiceMensalONI,
        AmpereIndiceTrimestralONI.name: AmpereIndiceTrimestralONI,
        AmperePrevClimaticaREE.name: AmperePrevClimaticaREE,
        AmperePrevClimaticaBacia.name: AmperePrevClimaticaBacia,
        AmpereEnaPrevREE.name: AmpereEnaPrevREE,
        UpdatedAmpereEnaPrevREE.name: UpdatedAmpereEnaPrevREE,
        AmpereEnaPrevBacia.name: AmpereEnaPrevBacia,
        AmpereZCAS.name: AmpereZCAS,
        AmpereASAS.name: AmpereASAS,
        AmpereJBN.name: AmpereJBN,
        AmpereFrentesFrias.name: AmpereFrentesFrias,
        AmpereIndiceMensalAMO.name: AmpereIndiceMensalAMO,
        AmpereIndiceMensalPDO.name: AmpereIndiceMensalPDO,
        AmpereIndiceMensalTNA.name: AmpereIndiceMensalTNA,
        AmpereIndiceMensalTSA.name: AmpereIndiceMensalTSA,
        AmpereIndiceMensalNINOS.name: AmpereIndiceMensalNINOS,
        AmperePrevClimaticaComplementoREE.name: AmperePrevClimaticaComplementoREE,
        AmperePrevClimaticaComplementoBacia.name: AmperePrevClimaticaComplementoBacia,
        AmperePrevClimaticaComplementoCPCREE.name: AmperePrevClimaticaComplementoCPCREE,
        AmperePrevClimaticaComplementoCPCBacia.name: AmperePrevClimaticaComplementoCPCBacia,
        AmperePrevClimaticaHistMensalENA.name: AmperePrevClimaticaHistMensalENA,
        AmperePrevClimaticaItem3e9_4REESUniao.name: AmperePrevClimaticaItem3e9_4REESUniao,
        AmperePrevClimaticaItem3e9_4BaciaUniao.name: AmperePrevClimaticaItem3e9_4BaciaUniao,
        CMOSemihorario.name: CMOSemihorario,
        AmpereIndiceMensalComplementoAMO.name: AmpereIndiceMensalComplementoAMO,
        AmpereIndiceMensalComplementoPDO.name: AmpereIndiceMensalComplementoPDO,
        AmpereIndiceMensalComplementoTNA.name: AmpereIndiceMensalComplementoTNA,
        AmpereIndiceMensalComplementoTSA.name: AmpereIndiceMensalComplementoTSA,
        AmpereIndiceMensalComplementoNINOS.name: AmpereIndiceMensalComplementoNINOS,
        AmpereIndiceMensalSAD.name: AmpereIndiceMensalSAD,
        AmpereIndiceMensalPrevisao24TNA.name: AmpereIndiceMensalPrevisao24TNA,
        AmpereIndiceMensalPrevisao24TSA.name: AmpereIndiceMensalPrevisao24TSA,
        AmpereIndiceMensalPrevisao24NINOS.name: AmpereIndiceMensalPrevisao24NINOS,
        AmpereEnaHistMensal.name: AmpereEnaHistMensal,
        AmpereClimatologiaEnaSUBs.name: AmpereClimatologiaEnaSUBs,
        AmpereEARdSubs.name: AmpereEARdSubs,
        PojecaoCargaRaw.name: PojecaoCargaRaw,
        AmpereEnaHistReeNew.name: AmpereEnaHistReeNew
    }

    def __get_table_class(self, table):
        try:
            return self.__TABLES[table]
        except KeyError:
            raise KeyError(f'"{table}" not found in Local tables. See Local().tables')
    
    def __init__(self):
        super().__init__()

    # retorna os nomes de todas as tabelas do banco de dados
    def __get_tables__(self):
        return list(self.__TABLES)

    def get_table(self, table: str, verbose=False) -> pd.DataFrame:
        return self.__get_table_class(table).get_data(self.proj_path)

    # retorna as colunas de uma determinada tabela
    def get_columns(self, table: str) -> pd.DataFrame:
        return self.__get_table_class(table).columns

    # retorna as colunas de uma determinada tabela
    def get_data_dict(self, table: str) -> pd.DataFrame:
        return self.__get_table_class(table).DATA_DICT

    # retorna as colunas de uma determinada tabela
    def get_type_dict(self, table: str) -> pd.DataFrame:
        return self.__get_table_class(table).DATA_TYPES_DICT

