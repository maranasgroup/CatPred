import { ref, computed } from 'vue'
import type { InputRow, Parameter, PredictionMode, SubstrateEntry } from '../api/types'

export interface InputRowEntry {
  id: number
  substrates: SubstrateEntry[]
  inhibitorSmiles: string
  sequence: string
  pdbpath: string
}

// Human glucokinase / hexokinase-4 (UniProt P35557) + D-glucose
// Related to paper example (Boorla & Maranas, Nat. Commun. 2025, Fig 7a)
const GCK_SEQUENCE =
  'MLDDRARMEAAKKEKVEQILAEFQLQEEDLKKVMRRMQKEMDRGLRLETHEEASVKMLPTYVRSTPEGS' +
  'EVGDFLSLDLGGTNFRVMLVKVGEGEEGQWSVKTKHQMYSIPEDAMTGTAEMLFDYISECISDFLDKHQM' +
  'KHKKLPLGFTFSFPVRHEDIDKGILLNWTKGFKASGAEGNNVVGLLRDAIKRRGDFEMDVVAMVNDTVAT' +
  'MISCYYEDHQCEVGMIVGTGCNACYMEEMQNVELVEGDEGRMCVNTEWGAFGDSGELDEFLLEYDRLVDE' +
  'SSANPGQQLYEKLIGGKYMGELVRLVLLRLVDENLLFHGEASEQLRTRGAFETRFVSQVESDTGDRKQIYN' +
  'ILSTLGLRPSTTDCDIVRRACESVSTRAAHMCSAGLAGVINRMRESRSEDVMRITVGVDGSVYKLHPSFKE' +
  'RFHASVRRLTPSCEITFIESEEGSGRGAALVSAVACKKACMLGQ'

// Classic enzyme: Human L-lactate dehydrogenase A (UniProt P00338) + pyruvate
const LDHA_SEQUENCE =
  'MATLKDQLIYNLLKEEQTPQNKITVVGVGAVGMACAISILMKDLADELALVDVIEDKLKGEMMDLQHGS' +
  'LFLRTPKIVSGKDYNVTANSKLVIITAGARQQEGESRLNLVQRNVNIFKFIIPNVVKYSPNCKLLIV' +
  'SNPVDILTYVAWKISGFPKNRVIGSGCNLDSARFRYLMGERLGVHPLSCHGWVLGEHGDSSVPVWSGMNV' +
  'AGVSLKTLHPDLGTDKDKEQWKEVHKQVVESAYEVIKLKGYTSWAIGLSVADLAESIMKNLRRVHPVSTM' +
  'IKGLYGIKDDVFLSVPCILGQNGISDLVKVTLTSEEEARLKKSADTLWGIQKELQF'

// Phenylalanine ammonia-lyase (UniProt P11544) — Ki sample enzyme
const P11544_SEQUENCE =
  'MAPSLDSISHSFANGVASAKQAVNGASTNLAVAGSHLPTTQVTQVDIVEKMLAAPTDSTLELDGYSLNLG' +
  'DVVSAARKGRPVRVKDSDEIRSKIDKSVEFLRSQLSMSVYGVTTGFGGSADTRTEDAISLQKALLEHQLCG' +
  'VLPSSFDSFRLGRGLENSLPLEVVRGAMTIRVNSLTRGHSAVRLVVLEALTNFLNHGITPIVPLRGTISAS' +
  'GDLSPLSYIAAAISGHPDSKVHVVHEGKEKILYAREAMALFNLEPVVLGPKEGLGLVNGTAVSASMATLA' +
  'LHDAHMLSLLSQSLTAMTVEAMVGHAGSFHPFLHDVTRPHPTQIEVAGNIRKLLEGSRFAVHHEEEVKVKD' +
  'DEGILRQDRYPLRTSPQWLGPLVSDLIHAHAVLTIEAGQSTTDNPLIDVENKTSHHGGNFQAAAVANTMEK' +
  'TRLGLAQIGKLNFTQLTEMLNAGMNRGLPSCLAAEDPSLSYHCKGLDIAAAAYTSELGHLANPVTTHVQPA' +
  'EMANQAVNSLALISARRTTESNDVLSLLLATHLYCVLQAIDLRAIEFEFKKQFGPAIVSLIDQHFGSAMTG' +
  'SNLRDELVEKVNKTLAKRLEQTNSYDLVPRWHDAFSFAAGTVVEVLSSTSLSLAAVNAWKVAAAESAISLT' +
  'RQVRETFWSAASTSSPALSYLSPRTQILYAFVREELGVKARRGDVFLGKQEVTIGSNVSKIYEAIKSGRINN' +
  'VLLKMLA'

const GLUCOSE_SMILES = 'C(C1C(C(C(C(O1)O)O)O)O)O'
const ATP_SMILES = 'C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N'
const PYRUVATE_SMILES = 'CC(=O)C(=O)O'
const NADH_SMILES = 'C1C=CN(C=C1C(=O)N)C2C(C(C(O2)COP(=O)(O)OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C(N=CN=C54)N)O)O)O)O'
const COUMARIC_ACID_SMILES = 'C1=CC(=CC=C1/C=C/C(=O)O)O'

const SAMPLE_SUBSTRATE: Omit<InputRowEntry, 'id'>[] = [
  {
    substrates: [
      { id: 1, smiles: GLUCOSE_SMILES, isPrimary: true },
      { id: 2, smiles: ATP_SMILES, isPrimary: false },
    ],
    inhibitorSmiles: '',
    sequence: GCK_SEQUENCE,
    pdbpath: 'GCK_HUMAN',
  },
  {
    substrates: [
      { id: 1, smiles: PYRUVATE_SMILES, isPrimary: true },
      { id: 2, smiles: NADH_SMILES, isPrimary: false },
    ],
    inhibitorSmiles: '',
    sequence: LDHA_SEQUENCE,
    pdbpath: 'LDHA_HUMAN',
  },
]

const SAMPLE_INHIBITION: Omit<InputRowEntry, 'id'>[] = [
  {
    substrates: [],
    inhibitorSmiles: COUMARIC_ACID_SMILES,
    sequence: P11544_SEQUENCE,
    pdbpath: 'P11544',
  },
]

// Validation helpers
const SMILES_VALID_CHARS = /^[A-Za-z0-9@+\-\[\]\\\/().=#%$:~&!*]+$/
const AMINO_ACIDS = new Set('ACDEFGHIKLMNPQRSTVWY'.split(''))

export function validateSmiles(s: string): string {
  if (!s.trim()) return 'SMILES is required.'
  if (!SMILES_VALID_CHARS.test(s.trim())) return 'Invalid SMILES characters.'
  return ''
}

export function validateSequence(s: string): string {
  if (!s.trim()) return 'Sequence is required.'
  const upper = s.trim().toUpperCase()
  for (const ch of upper) {
    if (!AMINO_ACIDS.has(ch)) return `Invalid amino acid: "${ch}".`
  }
  return ''
}

export function validatePdbpath(s: string): string {
  if (!s.trim()) return 'Sequence ID is required.'
  return ''
}

export function useInputRows() {
  let nextRowId = 1
  let nextSubId = 100
  const rows = ref<InputRowEntry[]>([])

  function formatSeqId(n: number): string {
    return `seq_${String(n).padStart(3, '0')}`
  }

  function getNextSeqId(): string {
    let maxSeen = 0
    for (const row of rows.value) {
      const match = row.pdbpath.match(/^seq_(\d+)$/i)
      if (match) {
        maxSeen = Math.max(maxSeen, Number(match[1]))
      }
    }
    if (maxSeen > 0) return formatSeqId(maxSeen + 1)
    return formatSeqId(rows.value.length + 1)
  }

  function makeSubstrate(smiles = '', isPrimary = false): SubstrateEntry {
    return { id: nextSubId++, smiles, isPrimary }
  }

  function addRow(values?: Partial<Omit<InputRowEntry, 'id'>>) {
    const substrates = values?.substrates?.length
      ? values.substrates.map((s) => makeSubstrate(s.smiles, s.isPrimary))
      : [makeSubstrate('', true)]

    rows.value.push({
      id: nextRowId++,
      substrates,
      inhibitorSmiles: values?.inhibitorSmiles ?? '',
      sequence: values?.sequence ?? '',
      pdbpath: values?.pdbpath || getNextSeqId(),
    })
  }

  function removeRow(id: number) {
    if (rows.value.length <= 1) return
    rows.value = rows.value.filter((r) => r.id !== id)
  }

  function updateField(
    id: number,
    field: 'sequence' | 'pdbpath' | 'inhibitorSmiles',
    value: string,
  ) {
    const row = rows.value.find((r) => r.id === id)
    if (row) {
      row[field] = value
    }
  }

  function addSubstrate(rowId: number) {
    const row = rows.value.find((r) => r.id === rowId)
    if (row) {
      row.substrates.push(makeSubstrate('', false))
    }
  }

  function removeSubstrate(rowId: number, subId: number) {
    const row = rows.value.find((r) => r.id === rowId)
    if (!row || row.substrates.length <= 1) return
    const wasPrimary = row.substrates.find((s) => s.id === subId)?.isPrimary
    row.substrates = row.substrates.filter((s) => s.id !== subId)
    if (wasPrimary && row.substrates.length > 0) {
      row.substrates[0].isPrimary = true
    }
  }

  function updateSubstrateSmiles(rowId: number, subId: number, smiles: string) {
    const row = rows.value.find((r) => r.id === rowId)
    if (!row) return
    const sub = row.substrates.find((s) => s.id === subId)
    if (sub) sub.smiles = smiles
  }

  function setPrimary(rowId: number, subId: number) {
    const row = rows.value.find((r) => r.id === rowId)
    if (!row) return
    for (const sub of row.substrates) {
      sub.isPrimary = sub.id === subId
    }
  }

  function loadSample(mode: PredictionMode) {
    rows.value = []
    nextRowId = 1
    nextSubId = 100
    const samples = mode === 'substrate' ? SAMPLE_SUBSTRATE : SAMPLE_INHIBITION
    for (const s of samples) {
      addRow(s)
    }
  }

  function clear() {
    rows.value = []
    nextRowId = 1
    nextSubId = 100
    addRow()
  }

  function importCsv(text: string, mode: PredictionMode): string {
    const lines = text.trim().split('\n')
    if (lines.length < 2) return 'CSV must have a header row and at least one data row.'

    const header = lines[0].split(',').map((h) => h.trim())
    const smilesIdx = header.findIndex((h) => h.toLowerCase() === 'smiles')
    const seqIdx = header.findIndex((h) => h.toLowerCase() === 'sequence')
    const pdbIdx = header.findIndex((h) => h.toLowerCase() === 'pdbpath')

    if (smilesIdx === -1 || seqIdx === -1) {
      return 'CSV must have SMILES and sequence columns.'
    }

    const newRows: Partial<Omit<InputRowEntry, 'id'>>[] = []
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',').map((c) => c.trim())
      if (!cols[smilesIdx] && !cols[seqIdx]) continue

      const smiles = cols[smilesIdx] || ''
      const entry: Partial<Omit<InputRowEntry, 'id'>> = {
        sequence: cols[seqIdx] || '',
        pdbpath: cols[pdbIdx] || formatSeqId(i),
      }

      if (mode === 'substrate') {
        const parts = smiles.split('.')
        entry.substrates = parts.map((s, idx) => ({
          id: idx + 1,
          smiles: s,
          isPrimary: idx === 0,
        }))
      } else {
        entry.inhibitorSmiles = smiles
      }

      newRows.push(entry)
    }

    if (newRows.length === 0) return 'No valid data rows found.'

    rows.value = []
    nextRowId = 1
    nextSubId = 100
    for (const r of newRows) {
      addRow(r)
    }
    return ''
  }

  function collectRowsForParameter(
    mode: PredictionMode,
    parameter: Parameter,
  ): InputRow[] {
    return rows.value
      .filter((r) => {
        if (mode === 'substrate') {
          return (
            r.substrates.some((s) => s.smiles.trim()) &&
            r.sequence.trim() &&
            r.pdbpath.trim()
          )
        }
        return r.inhibitorSmiles.trim() && r.sequence.trim() && r.pdbpath.trim()
      })
      .map((r) => {
        let smiles: string
        if (mode === 'inhibition') {
          smiles = r.inhibitorSmiles.trim()
        } else if (parameter === 'km') {
          const primary = r.substrates.find((s) => s.isPrimary)
          smiles = primary ? primary.smiles.trim() : r.substrates[0]?.smiles.trim() || ''
        } else {
          smiles = r.substrates
            .filter((s) => s.smiles.trim())
            .map((s) => s.smiles.trim())
            .join('.')
        }
        return {
          SMILES: smiles,
          sequence: r.sequence.trim().toUpperCase(),
          pdbpath: r.pdbpath.trim(),
        }
      })
  }

  function validateAll(mode: PredictionMode): string {
    if (rows.value.length === 0) return 'Please add at least one entry.'

    for (const row of rows.value) {
      if (mode === 'substrate') {
        if (!row.substrates.some((s) => s.smiles.trim())) {
          return 'Each entry needs at least one substrate SMILES.'
        }
        for (const sub of row.substrates) {
          if (sub.smiles.trim()) {
            const err = validateSmiles(sub.smiles)
            if (err) return err
          }
        }
      } else {
        const err = validateSmiles(row.inhibitorSmiles)
        if (err) return `Inhibitor: ${err}`
      }

      const seqErr = validateSequence(row.sequence)
      if (seqErr) return seqErr

      const idErr = validatePdbpath(row.pdbpath)
      if (idErr) return idErr
    }

    const collected =
      mode === 'substrate'
        ? collectRowsForParameter(mode, 'kcat')
        : collectRowsForParameter(mode, 'ki')

    const mapping = new Map<string, string>()
    for (const row of collected) {
      const existing = mapping.get(row.pdbpath)
      if (existing && existing !== row.sequence) {
        return 'Each Sequence ID must map to one unique enzyme sequence.'
      }
      mapping.set(row.pdbpath, row.sequence)
    }
    return ''
  }

  const rowCount = computed(() => rows.value.length)

  // Init with first substrate sample
  addRow(SAMPLE_SUBSTRATE[0])

  return {
    rows,
    rowCount,
    addRow,
    removeRow,
    updateField,
    addSubstrate,
    removeSubstrate,
    updateSubstrateSmiles,
    setPrimary,
    loadSample,
    clear,
    importCsv,
    collectRowsForParameter,
    validateAll,
  }
}
