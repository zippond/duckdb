#include "duckdb/execution/operator/schema/physical_create_art_index.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/execution/index/art/art_key.hpp"
#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/storage/table/append_state.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/execution/index/art/art_operator.hpp"
#include "duckdb/parser/parsed_data/alter_table_info.hpp"
#include "duckdb/parser/constraints/foreign_key_constraint.hpp"

namespace duckdb {

PhysicalCreateARTIndex::PhysicalCreateARTIndex(PhysicalPlan &physical_plan, LogicalOperator &op,
                                               TableCatalogEntry &table_p, const vector<column_t> &column_ids,
                                               unique_ptr<CreateIndexInfo> info,
                                               vector<unique_ptr<Expression>> unbound_expressions,
                                               idx_t estimated_cardinality, const bool sorted,
                                               unique_ptr<AlterTableInfo> alter_table_info)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::CREATE_INDEX, op.types, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info)), unbound_expressions(std::move(unbound_expressions)),
      sorted(sorted), alter_table_info(std::move(alter_table_info)) {

	// Convert the logical column ids to physical column ids.
	for (auto &column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(column_id)).index);
	}
	// Pre-build the BoundForeignKeyConstraint and full column type list for FK verification.
	if (this->alter_table_info && this->alter_table_info->IsAddForeignKey()) {
		auto &constraint_info = this->alter_table_info->Cast<AddConstraintInfo>();
		auto &fk = constraint_info.constraint->Cast<ForeignKeyConstraint>();
		physical_index_set_t pk_key_set(fk.info.pk_keys.begin(), fk.info.pk_keys.end());
		physical_index_set_t fk_key_set(fk.info.fk_keys.begin(), fk.info.fk_keys.end());
		bound_fk = make_uniq<BoundForeignKeyConstraint>(fk.info, std::move(pk_key_set), std::move(fk_key_set));
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//

class CreateARTIndexGlobalSinkState : public GlobalSinkState {
public:
	//! We merge the local indexes into one global index.
	unique_ptr<BoundIndex> global_index;
};

class CreateARTIndexLocalSinkState : public LocalSinkState {
public:
	explicit CreateARTIndexLocalSinkState(ClientContext &context) : arena_allocator(Allocator::Get(context)) {};

	unique_ptr<BoundIndex> local_index;
	ArenaAllocator arena_allocator;

	DataChunk key_chunk;
	unsafe_vector<ARTKey> keys;
	vector<column_t> key_column_ids;

	DataChunk row_id_chunk;
	unsafe_vector<ARTKey> row_ids;
};

unique_ptr<GlobalSinkState> PhysicalCreateARTIndex::GetGlobalSinkState(ClientContext &context) const {
	// Create the global sink state.
	auto state = make_uniq<CreateARTIndexGlobalSinkState>();

	// Create the global index.
	auto &storage = table.GetStorage();
	state->global_index = make_uniq<ART>(info->index_name, info->constraint_type, storage_ids,
	                                     TableIOManager::Get(storage), unbound_expressions, storage.db);
	return (std::move(state));
}

unique_ptr<LocalSinkState> PhysicalCreateARTIndex::GetLocalSinkState(ExecutionContext &context) const {
	// Create the local sink state and add the local index.
	auto state = make_uniq<CreateARTIndexLocalSinkState>(context.client);
	auto &storage = table.GetStorage();
	state->local_index = make_uniq<ART>(info->index_name, info->constraint_type, storage_ids,
	                                    TableIOManager::Get(storage), unbound_expressions, storage.db);

	// Initialize the local sink state.
	state->keys.resize(STANDARD_VECTOR_SIZE);
	state->row_ids.resize(STANDARD_VECTOR_SIZE);
	state->key_chunk.Initialize(Allocator::Get(context.client), state->local_index->logical_types);
	state->row_id_chunk.Initialize(Allocator::Get(context.client), vector<LogicalType> {LogicalType::ROW_TYPE});
	for (idx_t i = 0; i < state->key_chunk.ColumnCount(); i++) {
		state->key_column_ids.push_back(i);
	}
	return std::move(state);
}

SinkResultType PhysicalCreateARTIndex::SinkUnsorted(OperatorSinkInput &input) const {

	auto &l_state = input.local_state.Cast<CreateARTIndexLocalSinkState>();
	auto row_count = l_state.key_chunk.size();
	auto &art = l_state.local_index->Cast<ART>();

	// Insert each key and its corresponding row ID.
	for (idx_t i = 0; i < row_count; i++) {
		auto status = art.tree.GetGateStatus();
		auto conflict_type = ARTOperator::Insert(l_state.arena_allocator, art, art.tree, l_state.keys[i], 0,
		                                         l_state.row_ids[i], status, nullptr, IndexAppendMode::DEFAULT);
		D_ASSERT(conflict_type != ARTConflictType::TRANSACTION);
		if (conflict_type == ARTConflictType::CONSTRAINT) {
			throw ConstraintException("Data contains duplicates on indexed column(s)");
		}
	}

	return SinkResultType::NEED_MORE_INPUT;
}

SinkResultType PhysicalCreateARTIndex::SinkSorted(OperatorSinkInput &input) const {

	auto &l_state = input.local_state.Cast<CreateARTIndexLocalSinkState>();
	auto &storage = table.GetStorage();
	auto &l_index = l_state.local_index;

	// Construct an ART for this chunk.
	auto art = make_uniq<ART>(info->index_name, l_index->GetConstraintType(), l_index->GetColumnIds(),
	                          l_index->table_io_manager, l_index->unbound_expressions, storage.db,
	                          l_index->Cast<ART>().allocators);
	if (art->Build(l_state.keys, l_state.row_ids, l_state.key_chunk.size()) != ARTConflictType::NO_CONFLICT) {
		throw ConstraintException("Data contains duplicates on indexed column(s)");
	}

	// Merge the ART into the local ART.
	if (!l_index->MergeIndexes(*art)) {
		throw ConstraintException("Data contains duplicates on indexed column(s)");
	}

	return SinkResultType::NEED_MORE_INPUT;
}

SinkResultType PhysicalCreateARTIndex::Sink(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSinkInput &input) const {
	D_ASSERT(chunk.ColumnCount() >= 2);
	auto &l_state = input.local_state.Cast<CreateARTIndexLocalSinkState>();
	l_state.arena_allocator.Reset();
	l_state.key_chunk.ReferenceColumns(chunk, l_state.key_column_ids);

	// Check for NULLs, if we are creating a PRIMARY KEY.
	// FIXME: Later, we want to ensure that we skip the NULL check for any non-PK alter.
	if (alter_table_info && !bound_fk) {
		auto row_count = l_state.key_chunk.size();
		for (idx_t i = 0; i < l_state.key_chunk.ColumnCount(); i++) {
			if (VectorOperations::HasNull(l_state.key_chunk.data[i], row_count)) {
				throw ConstraintException("NOT NULL constraint failed: %s", info->index_name);
			}
		}
	}

	// Verify FK referential integrity for ALTER TABLE ADD FOREIGN KEY.
	// TODO: Does it verify local data? Add test to check this!
	if (bound_fk) {
		// key_chunk has FK columns at indices 0..N-1 (projected), but VerifyForeignKeyConstraint
		// uses fk_keys physical indices to address the chunk. Build a correctly-indexed chunk.
		vector<LogicalType> fk_table_types;
		for (auto &col : table.GetColumns().Physical()) {
			fk_table_types.emplace_back(col.Type());
		}
		DataChunk full_chunk;
		full_chunk.InitializeEmpty(fk_table_types);
		for (idx_t i = 0; i < bound_fk->info.fk_keys.size(); i++) {
			full_chunk.data[bound_fk->info.fk_keys[i].index].Reference(l_state.key_chunk.data[i]);
		}
		full_chunk.SetCardinality(l_state.key_chunk.size());
		table.GetStorage().VerifyFKReferentialIntegrity(*bound_fk, context.client, full_chunk);
	}

	l_state.local_index->Cast<ART>().GenerateKeyVectors(
	    l_state.arena_allocator, l_state.key_chunk, chunk.data[chunk.ColumnCount() - 1], l_state.keys, l_state.row_ids);

	if (sorted) {
		return SinkSorted(input);
	}
	return SinkUnsorted(input);
}

SinkCombineResultType PhysicalCreateARTIndex::Combine(ExecutionContext &context,
                                                      OperatorSinkCombineInput &input) const {
	auto &g_state = input.global_state.Cast<CreateARTIndexGlobalSinkState>();

	// Merge the local index into the global index.
	auto &l_state = input.local_state.Cast<CreateARTIndexLocalSinkState>();
	if (!g_state.global_index->MergeIndexes(*l_state.local_index)) {
		throw ConstraintException("Data contains duplicates on indexed column(s)");
	}

	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalCreateARTIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  OperatorSinkFinalizeInput &input) const {
	auto &state = input.global_state.Cast<CreateARTIndexGlobalSinkState>();

	// Vacuum excess memory and verify.
	state.global_index->Vacuum();
	D_ASSERT(!state.global_index->VerifyAndToString(true).empty());
	state.global_index->VerifyAllocations();

	auto &storage = table.GetStorage();
	if (!storage.IsMainTable()) {
		throw TransactionException(
		    "Transaction conflict: cannot add an index to a table that has been altered or dropped");
	}

	auto &schema = table.schema;
	info->column_ids = storage_ids;

	if (!alter_table_info) {
		// Ensure that the index does not yet exist in the catalog.
		auto entry = schema.GetEntry(schema.GetCatalogTransaction(context), CatalogType::INDEX_ENTRY, info->index_name);
		if (entry) {
			if (info->on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
				throw CatalogException("Index with name \"%s\" already exists!", info->index_name);
			}
			// IF NOT EXISTS on existing index. We are done.
			return SinkFinalizeType::READY;
		}

		auto index_entry = schema.CreateIndex(schema.GetCatalogTransaction(context), *info, table).get();
		D_ASSERT(index_entry);
		auto &index = index_entry->Cast<DuckIndexEntry>();
		index.initial_index_size = state.global_index->GetInMemorySize();

	} else {
		// Ensure that there are no other indexes with that name on this table.
		auto &indexes = storage.GetDataTableInfo()->GetIndexes();
		indexes.Scan([&](Index &index) {
			if (index.GetIndexName() == info->index_name) {
				throw CatalogException("an index with that name already exists for this table: %s", info->index_name);
			}
			return false;
		});

		auto &catalog = Catalog::GetCatalog(context, info->catalog);
		catalog.Alter(context, *alter_table_info);
	}

	// Add the index to the storage.
	storage.AddIndex(std::move(state.global_index));
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//

SourceResultType PhysicalCreateARTIndex::GetData(ExecutionContext &context, DataChunk &chunk,
                                                 OperatorSourceInput &input) const {
	return SourceResultType::FINISHED;
}

} // namespace duckdb
