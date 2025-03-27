import React from 'react';
import { format } from 'date-fns';
import { Task } from '../../types/task';
import Badge from '../ui/Badge';
import IconButton from '../ui/IconButton';
import { CheckCircleIcon, PencilIcon, TrashIcon } from '@heroicons/react/outline';

interface TaskCardProps {
  task: Task;
  onComplete: (id: string) => void;
  onEdit: (task: Task) => void;
  onDelete: (id: string) => void;
}

/**
 * Card component for displaying individual task information
 */
const TaskCard: React.FC<TaskCardProps> = ({ task, onComplete, onEdit, onDelete }) => {
  const { id, title, description, priority, status, category, dueDate } = task;
  
  // Priority color mapping
  const priorityColorMap = {
    high: 'text-red-600 bg-red-100',
    medium: 'text-yellow-600 bg-yellow-100',
    low: 'text-green-600 bg-green-100',
  };
  
  // Status color mapping
  const statusColorMap = {
    'not-started': 'text-gray-600 bg-gray-100',
    'in-progress': 'text-blue-600 bg-blue-100',
    completed: 'text-green-600 bg-green-100',
  };
  
  return (
    <div className={`border rounded-lg p-4 shadow-sm ${status === 'completed' ? 'bg-gray-50' : 'bg-white'}`}>
      <div className="flex items-start justify-between">
        <div className="flex items-center">
          <h3 className={`text-lg font-medium ${status === 'completed' ? 'line-through text-gray-500' : 'text-gray-900'}`}>
            {title}
          </h3>
        </div>
        <div className="flex space-x-2">
          <Badge 
            label={priority} 
            className={priorityColorMap[priority as keyof typeof priorityColorMap]} 
          />
          <Badge 
            label={status.replace('-', ' ')} 
            className={statusColorMap[status as keyof typeof statusColorMap]} 
          />
        </div>
      </div>
      
      {description && (
        <p className={`mt-2 text-sm ${status === 'completed' ? 'text-gray-400' : 'text-gray-600'}`}>
          {description}
        </p>
      )}
      
      <div className="mt-4 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-500">
            {category.name}
          </span>
          {dueDate && (
            <span className={`text-sm ${
              new Date(dueDate) < new Date() && status !== 'completed'
                ? 'text-red-600'
                : 'text-gray-500'
            }`}>
              Due {format(new Date(dueDate), 'MMM dd, yyyy')}
            </span>
          )}
        </div>
        
        <div className="flex space-x-2">
          {status !== 'completed' && (
            <IconButton
              icon={<CheckCircleIcon className="h-5 w-5" />}
              onClick={() => onComplete(id)}
              aria-label="Complete task"
              variant="success"
            />
          )}
          <IconButton
            icon={<PencilIcon className="h-5 w-5" />}
            onClick={() => onEdit(task)}
            aria-label="Edit task"
            variant="secondary"
          />
          <IconButton
            icon={<TrashIcon className="h-5 w-5" />}
            onClick={() => onDelete(id)}
            aria-label="Delete task"
            variant="danger"
          />
        </div>
      </div>
    </div>
  );
};

export default TaskCard;