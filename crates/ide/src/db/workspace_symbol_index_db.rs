use std::{collections::HashMap, hash::Hash, ops::Deref, sync::Mutex};

use base_db::{
    salsa::{Revision, plumbing::current_revision},
    source_db::SourceRootDb,
    source_root::SourceRootId,
};
use hir_def::def_id::DefId;
use hir_ty::db::TyDb;
use triomphe::Arc;
use vfs::FileId;

use crate::{
    ScopeVisibility,
    semantic_index::{
        FileModuleEdges, FileModuleIndex, FileSemanticIndex, ModuleIndex, SemanticIndex,
    },
    workspace_symbols::{SymbolIndex, WorkspaceSymbol},
};

#[salsa::db]
pub trait WorkspaceSymbolIndexDb: SourceRootDb + TyDb {
    fn index_cache(&self) -> &IndexCache;
}

/// Lazily built indexes shared by database clones for one Salsa revision.
#[derive(Default)]
pub struct IndexCache(Mutex<RevisionIndexes>);

#[derive(Default)]
struct RevisionIndexes {
    revision: Option<Revision>,
    modules: HashMap<SourceRootId, Arc<ModuleIndex>>,
    files: HashMap<FileId, Arc<FileSemanticIndex>>,
}

fn cached<K: Eq + Hash, V>(
    db: &dyn WorkspaceSymbolIndexDb,
    select: impl Fn(&mut RevisionIndexes) -> &mut HashMap<K, Arc<V>>,
    key: K,
    build: impl FnOnce() -> Arc<V>,
) -> Arc<V> {
    // A cache hit would otherwise skip the usual Salsa cancellation check.
    db.unwind_if_revision_cancelled();
    let revision = current_revision(db);
    let cache = &db.index_cache().0;
    let hit = {
        let mut indexes = cache.lock().unwrap();
        if indexes.revision != Some(revision) {
            // An index may depend on other files, includes, and project inputs.
            *indexes = RevisionIndexes { revision: Some(revision), ..Default::default() };
        }
        select(&mut indexes).get(&key).cloned()
    };
    if let Some(value) = hit {
        return value;
    }
    // Build without the lock, since Salsa may block or unwind. Concurrent
    // misses can build twice, but only one result is retained. Salsa cannot
    // advance the revision while this database handle is in use.
    let value = build();
    let mut indexes = cache.lock().unwrap();
    select(&mut indexes).entry(key).or_insert(value).clone()
}

// Expose the lower Salsa query surface without rebuilding it as IDE wrappers.
impl Deref for dyn WorkspaceSymbolIndexDb {
    type Target = dyn TyDb;

    fn deref(&self) -> &Self::Target {
        self
    }
}

impl dyn WorkspaceSymbolIndexDb + '_ {
    pub fn file_workspace_symbols(&self, file_id: FileId) -> Arc<[WorkspaceSymbol]> {
        file_workspace_symbols(self, file_id)
    }

    pub fn source_root_symbol_index(&self, source_root_id: SourceRootId) -> Arc<SymbolIndex> {
        source_root_symbol_index(self, source_root_id)
    }

    pub fn source_root_module_index(&self, source_root_id: SourceRootId) -> Arc<ModuleIndex> {
        source_root_module_index(self, source_root_id)
    }

    pub fn source_root_semantic_index(&self, source_root_id: SourceRootId) -> Arc<SemanticIndex> {
        source_root_semantic_index(self, source_root_id)
    }

    pub fn file_module_index(&self, file_id: FileId) -> Arc<FileModuleIndex> {
        file_module_index(self, file_id)
    }

    pub fn file_module_edges(&self, file_id: FileId) -> Arc<FileModuleEdges> {
        file_module_edges(self, file_id)
    }

    pub fn file_semantic_index(&self, file_id: FileId) -> Arc<FileSemanticIndex> {
        file_semantic_index(self, file_id)
    }

    /// Distinct source roots derived from the current file set, in stable
    /// order. Module-name resolution scans every root's module index, so both
    /// callers (`module_candidates`, `module_edges`) share one implementation
    /// instead of each recomputing `files().map(source_root_id)` inline.
    pub fn workspace_source_root_ids(&self) -> Vec<SourceRootId> {
        let mut ids =
            self.files().iter().map(|&file_id| self.source_root_id(file_id)).collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// The connected component of same-name port connections around `def`,
    /// in discovery order. Shared by the recursive rename info, conflict and
    /// edit commands so a single F2 interaction computes it once.
    pub fn recursive_rename_closure(
        &self,
        def: DefId,
        visibility: ScopeVisibility,
        single_file: Option<FileId>,
    ) -> Arc<Vec<DefId>> {
        recursive_rename_closure(self, def, visibility, single_file)
    }
}

fn file_workspace_symbols(
    db: &dyn WorkspaceSymbolIndexDb,
    file_id: FileId,
) -> Arc<[WorkspaceSymbol]> {
    crate::workspace_symbols::file_symbols(db, file_id)
}

fn source_root_symbol_index(
    db: &dyn WorkspaceSymbolIndexDb,
    source_root_id: SourceRootId,
) -> Arc<SymbolIndex> {
    Arc::new(SymbolIndex::for_source_root(db, source_root_id))
}

fn source_root_module_index(
    db: &dyn WorkspaceSymbolIndexDb,
    source_root_id: SourceRootId,
) -> Arc<ModuleIndex> {
    cached(
        db,
        |indexes| &mut indexes.modules,
        source_root_id,
        || Arc::new(ModuleIndex::for_source_root(db, source_root_id)),
    )
}

fn source_root_semantic_index(
    db: &dyn WorkspaceSymbolIndexDb,
    source_root_id: SourceRootId,
) -> Arc<SemanticIndex> {
    Arc::new(SemanticIndex::for_source_root(db, source_root_id))
}

pub(crate) fn source_root_symbol_index_for_root(
    db: &dyn WorkspaceSymbolIndexDb,
    source_root_id: SourceRootId,
) -> Arc<SymbolIndex> {
    db.source_root_symbol_index(source_root_id)
}

pub(crate) fn source_root_module_index_for_root(
    db: &dyn WorkspaceSymbolIndexDb,
    source_root_id: SourceRootId,
) -> Arc<ModuleIndex> {
    db.source_root_module_index(source_root_id)
}

pub(crate) fn source_root_semantic_index_for_root(
    db: &dyn WorkspaceSymbolIndexDb,
    source_root_id: SourceRootId,
) -> Arc<SemanticIndex> {
    db.source_root_semantic_index(source_root_id)
}

fn file_module_index(db: &dyn WorkspaceSymbolIndexDb, file_id: FileId) -> Arc<FileModuleIndex> {
    Arc::new(crate::semantic_index::FileModuleIndex::for_file(db, file_id))
}

fn file_module_edges(db: &dyn WorkspaceSymbolIndexDb, file_id: FileId) -> Arc<FileModuleEdges> {
    Arc::new(crate::semantic_index::FileModuleEdges::for_file(db, file_id))
}

fn file_semantic_index(db: &dyn WorkspaceSymbolIndexDb, file_id: FileId) -> Arc<FileSemanticIndex> {
    cached(
        db,
        |indexes| &mut indexes.files,
        file_id,
        || Arc::new(FileSemanticIndex::for_file(db, file_id)),
    )
}

fn recursive_rename_closure(
    db: &dyn WorkspaceSymbolIndexDb,
    def: DefId,
    visibility: ScopeVisibility,
    single_file: Option<FileId>,
) -> Arc<Vec<DefId>> {
    Arc::new(crate::rename::recursive_rename_closure_impl(db, def, visibility, single_file))
}
