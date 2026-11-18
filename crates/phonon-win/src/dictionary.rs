//! The shared macOS dictionary schema, stored under the Windows data root.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context, Result};
use phonon_core::data::{DictionaryEntry, DictionaryFile};

const SEED_PHRASES: [&str; 9] = [
    "Phonon",
    "Droid",
    "Factory",
    "Factory Desktop",
    "WSL",
    "GB",
    "cargo",
    "PostgreSQL",
    "GitHub",
];

pub fn load() -> Result<DictionaryFile> {
    load_at(&crate::paths::data_root().join("dictionary.json"))
}

fn load_at(path: &Path) -> Result<DictionaryFile> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            seed_if_missing(path)?;
            fs::read(path).with_context(|| format!("read {}", path.display()))?
        }
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    serde_json::from_slice(&bytes).with_context(|| format!("parse dictionary {}", path.display()))
}

fn seeded() -> DictionaryFile {
    DictionaryFile {
        entries: SEED_PHRASES
            .iter()
            .map(|phrase| DictionaryEntry {
                phrase: (*phrase).to_string(),
                replacement: None,
                spoken_forms: Vec::new(),
                source: "phonon-win-seed".into(),
                starred: false,
                usage_count: 0,
            })
            .collect(),
        ..Default::default()
    }
}

fn seed_if_missing(path: &Path) -> Result<()> {
    let parent = path.parent().context("dictionary path has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create temporary dictionary in {}", parent.display()))?;
    serde_json::to_writer_pretty(temporary.as_file_mut(), &seeded())
        .with_context(|| format!("write seed dictionary for {}", path.display()))?;
    temporary
        .as_file()
        .sync_all()
        .with_context(|| format!("sync seed dictionary for {}", path.display()))?;
    // On Windows this renames without REPLACE_EXISTING. A concurrent startup
    // or user edit wins; never replace a file that appeared after our read.
    match temporary.persist_noclobber(path) {
        Ok(_) => Ok(()),
        Err(error) if error.error.kind() == ErrorKind::AlreadyExists => Ok(()),
        Err(error) => {
            Err(error.error).with_context(|| format!("create dictionary {}", path.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polish::{dictionary_user_message, transcript_payload, user_message};

    #[test]
    fn missing_dictionary_is_seeded_in_the_shared_schema() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("data/dictionary.json");
        let dictionary = load_at(&path).unwrap();
        assert_eq!(dictionary.schema_version, 1);
        assert!(dictionary.updated_at_unix_ms > 0);
        assert_eq!(
            dictionary
                .entries
                .iter()
                .map(|entry| entry.phrase.as_str())
                .collect::<Vec<_>>(),
            SEED_PHRASES
        );
        for entry in &dictionary.entries {
            assert!(entry.spoken_forms.is_empty());
            assert_eq!(entry.source, "phonon-win-seed");
            assert!(!entry.starred);
            assert_eq!(entry.usage_count, 0);
            assert!(entry.replacement.is_none());
        }
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            serde_json::from_slice::<DictionaryFile>(&bytes).unwrap(),
            dictionary
        );
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["entries"]
            .as_array()
            .unwrap()
            .iter()
            .all(|entry| entry.get("replacement").is_none()));
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn existing_dictionary_is_not_overwritten_even_if_it_appears_before_publish() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("dictionary.json");
        let bytes = br#"{"schema_version":1,"updated_at_unix_ms":123,"entries":[{"phrase":"wip","spoken_forms":["work in progress"],"source":"user","starred":true,"usage_count":7,"replacement":"WIP"}]}
"#;
        fs::write(&path, bytes).unwrap();
        let dictionary = load_at(&path).unwrap();
        assert_eq!(dictionary.entries[0].phrase, "wip");
        assert_eq!(dictionary.entries[0].canonical(), "WIP");
        seed_if_missing(&path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(load_at(&path).unwrap(), dictionary);
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
    }

    #[test]
    fn malformed_dictionary_names_the_path_and_preserves_user_edits() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("dictionary.json");
        for contents in [
            "{not json",
            r#"{"schema_version":1,"updated_at_unix_ms":0,"entries":[{"phrase":7}]}"#,
        ] {
            fs::write(&path, contents).unwrap();
            let error = load_at(&path).unwrap_err().to_string();
            assert!(error.contains("parse dictionary"), "{error}");
            assert!(error.contains(&path.display().to_string()), "{error}");
            assert_eq!(fs::read_to_string(&path).unwrap(), contents);
        }
    }

    #[test]
    fn loaded_dictionary_is_a_snapshot_until_the_next_load() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("dictionary.json");
        let loaded = load_at(&path).unwrap();
        let empty = DictionaryFile::default();
        fs::write(&path, serde_json::to_vec(&empty).unwrap()).unwrap();
        assert_eq!(loaded.entries.len(), SEED_PHRASES.len());
        assert_eq!(load_at(&path).unwrap(), empty);
    }

    #[test]
    fn empty_dictionary_keeps_the_original_user_message_bytes() {
        let dictionary = DictionaryFile::default();
        for raw in ["", " \t\r\n", "  hello there  ", "\r\nHello, 世界!\t"] {
            assert_eq!(
                dictionary_user_message(&dictionary, raw).as_bytes(),
                user_message(raw).as_bytes()
            );
        }
    }

    #[test]
    fn seeded_dictionary_encloses_the_transcript_and_relevant_terms() {
        let raw = "open droid on wsl";
        let message = dictionary_user_message(&seeded(), raw);
        assert!(message.starts_with("<phonon_dictionary>\n"), "{message}");
        assert!(
            message.contains("</phonon_dictionary>\n<transcript>\n"),
            "{message}"
        );
        assert_eq!(transcript_payload(&message), raw);
        let terms = message
            .lines()
            .find_map(|line| line.strip_prefix("canonical_terms: "))
            .unwrap();
        let terms = terms.split(", ").collect::<Vec<_>>();
        assert!(terms.contains(&"Droid"), "{message}");
        assert!(terms.contains(&"WSL"), "{message}");
    }

    #[test]
    fn seeded_dictionary_preserves_on_between_canonical_terms() {
        let raw = "open Phonon on GitHub";
        let message = dictionary_user_message(&seeded(), raw);
        assert_eq!(transcript_payload(&message), raw);
    }

    #[test]
    fn ordinary_phone_is_not_precorrected_to_phonon() {
        // Windows embeds the shared English list, so the active ordinary-word
        // guard must protect "phone" while still allowing nonword term repairs.
        let raw = "my phone is in the car";
        let message = dictionary_user_message(&seeded(), raw);
        assert_eq!(transcript_payload(&message), raw);
    }
}
